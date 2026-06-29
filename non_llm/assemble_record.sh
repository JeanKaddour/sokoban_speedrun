#!/usr/bin/env bash
# Assemble a non-LLM leaderboard record — or its verification rerun — from a finished run.
#
# Reads from a LOCAL run dir by default (SOURCE=local); set SOURCE=modal to pull off the volume.
# A single train call runs the final held-out eval in the same shot, so the run log, eval JSON, and
# source snapshot all land under the run dir (eval named eval_step<N>_seed<EVAL_SEED>.json):
#   LOCAL  : uv run python speedrun.py --run <RUN> ...                 -> outputs/<RUN>/
#   MODAL  : RUN_NAME=<RUN> DIFFICULTY=4 TOTAL_TIMESTEPS=<steps> uv run modal run --detach modal_app_non_llm.py
#            (re-eval an existing checkpoint: EVAL_CHECKPOINT=/vol/outputs/<RUN>/final.pt RUN_NAME=<RUN> ...)
#
# SUBMISSION (the record itself):
#   RUN=<RUN> DEST=records/<date>_01_<name> IDEA_FILE=../reports/non-llm-idea.md ./assemble_record.sh
#
# VERIFICATION (an independent rerun, dropped into the record's verification/ subdir — run a SECOND
# train+eval with a different seed first (--seed <vseed>), then):
#   RUN=<VRUN> VERIFY_OF=records/<date>_01_<name> ./assemble_record.sh
#   (seed is read from the run log; add VERIFIER="@me  PR#12" to attribute, or TRAIN_SEED= to override)
#
# Collects only the record artifacts (not the multi-MB checkpoint): the run log, eval JSON, and source
# snapshot, renamed to the seed convention, then regenerates the record report + verifies
# (verify_record checks submission AND verification). A submission also inserts/refreshes this record's
# README leaderboard row and redraws the figures (fill in Description + Contributors after).
set -euo pipefail

RUN="${RUN:?set RUN to the run name (local outputs/<RUN>/, or outputs/<RUN>/ on the volume with SOURCE=modal)}"
TRAIN_SEED="${TRAIN_SEED:-}"     # file label; auto-derived from the run log's args if unset
EVAL_SEED="${EVAL_SEED:-12345}"  # pinned eval seed (the JSON's sampling.seed / eval_step<...>_seed<EVAL_SEED>)
STEP="${STEP:-}"                 # optional eval checkpoint step; inferred from eval artifacts if unset
SOURCE="${SOURCE:-local}"        # local | modal — where the run artifacts live
LOCAL_OUTPUTS="${LOCAL_OUTPUTS:-outputs}"  # base dir for local runs (SOURCE=local)
VOL="${VOL:-nanochat-rl-hf}"     # shared with the LLM track (see modal_app_non_llm.VOLUME_NAME)
IDEA_FILE="${IDEA_FILE:-}"
VERIFY_OF="${VERIFY_OF:-}"       # if set, assemble as the verification of this record dir
VERIFIER="${VERIFIER:-}"         # one-line attribution written to verification/verifier.txt

if [ -n "$VERIFY_OF" ]; then
  RECORD="$VERIFY_OF"; DEST="$VERIFY_OF/verification"
else
  DEST="${DEST:?set DEST to the record dir, e.g. records/2026-06-21_01_non_llm}"; RECORD="$DEST"
fi

stage="$(mktemp -d)"
trap 'rm -rf "$stage"' EXIT

normalize_step() {
  local raw="$1"
  raw="${raw#step_}"
  raw="${raw#step}"
  [[ "$raw" =~ ^[0-9]+$ ]] || { echo "ERROR: STEP must be numeric or step_<n>, got '$1'"; exit 1; }
  printf "%06d" "$((10#$raw))"
}

[[ "$EVAL_SEED" =~ ^[0-9]+$ ]] || { echo "ERROR: EVAL_SEED must be numeric, got '$EVAL_SEED'"; exit 1; }

# Local and Modal share the same run-dir layout (eval_step<N>_seed<EVAL_SEED>.json beside the log +
# source/), so discovery is one listing + a fetch helper that either copies or pulls off the volume.
if [ "$SOURCE" = modal ]; then
  echo ">> volume listing /outputs/$RUN"
  listing="$(modal volume ls "$VOL" "/outputs/$RUN")"
else
  run_dir="$LOCAL_OUTPUTS/$RUN"
  [ -d "$run_dir" ] || {
    echo "ERROR: local run dir '$run_dir' not found (set LOCAL_OUTPUTS=<dir>, or SOURCE=modal to pull off the volume)"
    exit 1
  }
  echo ">> local run dir: $run_dir"
  listing="$(ls -1 "$run_dir")"
fi
fetch() {  # <relpath under the run dir> -> $stage/ (preserving subdirs)
  mkdir -p "$stage/$(dirname "$1")"
  if [ "$SOURCE" = modal ]; then
    modal volume get "$VOL" "/outputs/$RUN/$1" "$stage/$1" --force
  else
    cp "$LOCAL_OUTPUTS/$RUN/$1" "$stage/$1"
  fi
}

printf '%s\n' "$listing"
log_name="$(printf '%s\n' "$listing" | grep -oE 'log_[0-9a-f]{8}\.txt' | head -1 || true)"
[ -n "$log_name" ] || { echo "ERROR: no log_*.txt under $SOURCE run $RUN"; exit 1; }
echo ">> run log: $log_name"

if [ -n "$STEP" ]; then
  STEP="$(normalize_step "$STEP")"
else
  eval_json="$(printf '%s\n' "$listing" \
    | grep -oE "eval_step[0-9]+_seed${EVAL_SEED}\.json" \
    | sort -V \
    | tail -1 || true)"
  [ -n "$eval_json" ] || {
    echo "ERROR: no eval_step*_seed${EVAL_SEED}.json under $SOURCE run $RUN"
    echo "       Run the final-checkpoint eval first, or set STEP=<final checkpoint step>."
    exit 1
  }
  STEP="${eval_json#eval_step}"
  STEP="${STEP%%_seed*}"
fi
eval_json="eval_step${STEP}_seed${EVAL_SEED}.json"
echo ">> eval artifact: $eval_json"

echo ">> collecting record artifacts (not the checkpoint)"
fetch "$log_name"
fetch "$eval_json"
mkdir -p "$stage/source"
if fetch "source/speedrun.py" 2>/dev/null; then
  echo ">> got source snapshot"
else
  echo ">> source snapshot not found in run output; report generation will backfill from train log"
fi

# Label files with the run's actual training seed (from the log's args attestation) unless TRAIN_SEED
# was set explicitly. Verification just needs a different seed than the submission — no flag required.
if [ -z "$TRAIN_SEED" ]; then
  TRAIN_SEED="$(grep -m1 '^args: {' "$stage/$log_name" | grep -oE "'seed': *[0-9]+" | grep -oE '[0-9]+' || true)"
  if [ -n "$TRAIN_SEED" ]; then
    echo ">> train seed (from log args): $TRAIN_SEED"
  else
    TRAIN_SEED=42
    echo ">> WARN: no seed in $log_name args; labeling files seed $TRAIN_SEED (override with TRAIN_SEED=)"
  fi
fi

mkdir -p "$DEST"
cp "$stage/$log_name"  "$DEST/train_log_seed${TRAIN_SEED}.txt"
cp "$stage/$eval_json" "$DEST/eval_seed${TRAIN_SEED}.json"
if [ -f "$stage/source/speedrun.py" ]; then
  mkdir -p "$DEST/source"
  cp "$stage/source/speedrun.py" "$DEST/source/speedrun_seed${TRAIN_SEED}.py"
  if [ -z "$VERIFY_OF" ]; then
    # Pin the canonical recipe to this submission (modded-nanogpt / slowrun convention):
    # the top-level speedrun.py always holds the current record's code. Lands in the PR diff.
    cp "$stage/source/speedrun.py" speedrun.py
    echo ">> pinned ./speedrun.py to this record's recipe (review the diff — it's part of your PR)"
  fi
elif [ -z "$VERIFY_OF" ]; then
  echo ">> WARN: no source snapshot in run output — ./speedrun.py NOT pinned; commit the recipe you ran by hand"
fi
echo ">> assembled $DEST (seed $TRAIN_SEED)"

if [ -n "$VERIFY_OF" ]; then
  [ -n "$VERIFIER" ] && printf '%s\n' "$VERIFIER" > "$DEST/verifier.txt"
  echo ">> verifier.txt: ${VERIFIER:-<unset — add one before merging>}"
elif [ -n "$IDEA_FILE" ] && [ ! -f "$DEST/README.md" ]; then
  { echo "# $(basename "$DEST")"; echo; cat "$IDEA_FILE"; } > "$DEST/README.md"
  echo ">> seeded $DEST/README.md from $IDEA_FILE (auto block appended by make_record_report)"
fi

# A submission also writes its README leaderboard row + redraws the figures; a verification doesn't
# (it reruns the same record, adding no new row).
report_args=("$RECORD" --track non-llm)
[ -z "$VERIFY_OF" ] && report_args+=(--update-leaderboard)
echo ">> generating report + plots for $RECORD"
uv run python ../make_record_report.py "${report_args[@]}"
echo ">> verifying $RECORD (submission + verification)"
uv run python verify_record.py "$RECORD"
echo
if [ -n "$VERIFY_OF" ]; then
  echo ">> DONE. Verification appended to $RECORD; its README now has a ## Verification section."
else
  echo ">> DONE. Leaderboard row added + figures redrawn. Next:"
  echo "   1) fill in the new row's Description + Contributors in README.md"
  echo "   2) run a verification rerun (different seed): RUN=<VRUN> VERIFY_OF=$DEST ./assemble_record.sh"
  echo "   3) remove any superseded record dir, then open the PR"
fi
