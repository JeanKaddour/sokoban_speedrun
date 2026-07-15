#!/usr/bin/env bash
# Assemble a leaderboard record — or its verification rerun — from a finished run.
#
# Reads from a LOCAL run dir by default (SOURCE=local); set SOURCE=modal to pull off the volume.
# Both modes need the train run AND its final-checkpoint eval (the LLM track runs eval as a second
# command). Local layout (the default, written by torchrun + eval_speedrun):
#   NODE_GPUS=8 uv run torchrun --standalone --nproc_per_node=3 -m speedrun -- --run <RUN> ...
#   uv run python -m eval_speedrun --run <RUN>-final-eval --eval-checkpoint outputs/<RUN>/step_<N> \
#       --eval-seed 12345 ...
#   # -> outputs/<RUN>/ (log + source) and outputs/<RUN>-final-eval/ (eval_step<N>.json + rollouts)
# Modal layout (SOURCE=modal): everything under /outputs/<RUN>/ on the volume, eval named
# eval_step<N>_seed<EVAL_SEED>.json (set RUN_NAME=<RUN> on both `modal run` calls).
#
# SUBMISSION (the record itself):
#   RUN=<RUN> DEST=records/<date>_01_<name> ./assemble_record.sh
# The report is scaffolded with a placeholder "## Idea" section — fill it in by hand afterwards.
#
# VERIFICATION (an independent rerun, dropped into the record's verification/ subdir — run a SECOND
# training+eval with a different seed first, then):
#   RUN=<VRUN> VERIFY_OF=records/<date>_01_<name> ./assemble_record.sh
#   (seed is read from the run log; add VERIFIER="@me  PR#12" to attribute, or TRAIN_SEED= to override)
#
# Collects only the record artifacts (not the multi-GB checkpoint), including the source snapshot
# when present, renames to the seed convention, then regenerates the record report + verifies
# (verify_record checks submission AND verification). A submission also inserts/refreshes this
# record's README leaderboard row and redraws the leaderboard + rolling training figures
# (fill in Description + Contributors after). Verification refreshes those assets with seed two.
set -euo pipefail

RUN="${RUN:?set RUN to the run name (local outputs/<RUN>/, or outputs/<RUN>/ on the volume with SOURCE=modal)}"
TRAIN_SEED="${TRAIN_SEED:-}"     # file label; auto-derived from the run log's args if unset
EVAL_SEED="${EVAL_SEED:-12345}"  # pinned eval sampling seed (the JSON's sampling.seed / "seed")
STEP="${STEP:-}"                 # optional eval checkpoint step; inferred from eval artifacts if unset
SOURCE="${SOURCE:-local}"        # local | modal — where the run artifacts live
LOCAL_OUTPUTS="${LOCAL_OUTPUTS:-outputs}"  # base dir for local runs (SOURCE=local)
EVAL_JSON="${EVAL_JSON:-}"       # SOURCE=local: explicit eval JSON path, overriding auto-detect
VOL="${VOL:-nanochat-rl-hf}"
VERIFY_OF="${VERIFY_OF:-}"       # if set, assemble as the verification of this record dir
VERIFIER="${VERIFIER:-}"         # one-line attribution written to verification/verifier.txt

if [ -n "$VERIFY_OF" ]; then
  RECORD="$VERIFY_OF"; DEST="$VERIFY_OF/verification"
else
  DEST="${DEST:?set DEST to the record dir, e.g. records/2026-06-14_01_baseline}"; RECORD="$DEST"
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

eval_roll=""  # set below if a rollouts artifact is present (verify re-scores it when so)

if [ "$SOURCE" = modal ]; then
  echo ">> volume listing /outputs/$RUN"
  volume_listing="$(modal volume ls "$VOL" "/outputs/$RUN")"
  printf '%s\n' "$volume_listing"
  log_name="$(printf '%s\n' "$volume_listing" | grep -oE 'log_[0-9a-f]{8}\.txt' | head -1 || true)"
  [ -n "$log_name" ] || { echo "ERROR: no log_*.txt under /outputs/$RUN"; exit 1; }
  echo ">> run log: $log_name"

  if [ -n "$STEP" ]; then
    STEP="$(normalize_step "$STEP")"
  else
    eval_json="$(printf '%s\n' "$volume_listing" \
      | grep -oE "eval_step[0-9]+_seed${EVAL_SEED}\.json" \
      | sort -V \
      | tail -1 || true)"
    [ -n "$eval_json" ] || {
      echo "ERROR: no eval_step*_seed${EVAL_SEED}.json under /outputs/$RUN"
      echo "       Run the final-checkpoint eval first, or set STEP=<final checkpoint step>."
      exit 1
    }
    STEP="${eval_json#eval_step}"
    STEP="${STEP%%_seed*}"
  fi
  eval_json="eval_step${STEP}_seed${EVAL_SEED}.json"
  eval_roll="eval_step${STEP}_seed${EVAL_SEED}.rollouts.jsonl.gz"
  echo ">> eval artifact: $eval_json"

  # Verification needs only log + eval (+rollouts); a submission also keeps final_rollouts.
  pull=("$log_name" "$eval_json" "$eval_roll")
  [ -z "$VERIFY_OF" ] && pull+=(final_rollouts.jsonl.gz)
  echo ">> pulling record artifacts (not the checkpoint)"
  for f in "${pull[@]}"; do
    modal volume get "$VOL" "/outputs/$RUN/$f" "$stage/" --force
  done
  mkdir -p "$stage/source"
  if modal volume get "$VOL" "/outputs/$RUN/source/speedrun.py" "$stage/source/" --force; then
    echo ">> pulled source snapshot"
  else
    echo ">> source snapshot not found in run output; report generation will backfill from train log"
  fi
  if modal volume get "$VOL" "/outputs/$RUN/metrics.jsonl" "$stage/" --force; then
    echo ">> pulled metrics.jsonl"
  else
    echo ">> metrics.jsonl not found in run output (pre-metrics run); plots fall back to step: lines"
  fi
else
  run_dir="$LOCAL_OUTPUTS/$RUN"
  [ -d "$run_dir" ] || {
    echo "ERROR: local run dir '$run_dir' not found (set LOCAL_OUTPUTS=<dir>, or SOURCE=modal to pull off the volume)"
    exit 1
  }
  echo ">> local run dir: $run_dir"
  log_path="$(ls -1 "$run_dir"/log_*.txt 2>/dev/null | head -1 || true)"
  [ -n "$log_path" ] || { echo "ERROR: no log_*.txt in $run_dir"; exit 1; }
  log_name="$(basename "$log_path")"
  echo ">> run log: $log_name"

  # Eval usually lands in a sibling dir (e.g. <RUN>-final-eval) named eval_step<N>.json with no seed
  # suffix; pick the newest one whose internal "seed" matches EVAL_SEED. Override with EVAL_JSON=<path>.
  if [ -n "$EVAL_JSON" ]; then
    eval_src="$EVAL_JSON"
  else
    eval_src="$(
      for f in "$LOCAL_OUTPUTS/$RUN"*/eval_step*.json "$run_dir"/eval_*.json; do
        [ -f "$f" ] || continue
        s="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("seed",""))' "$f" 2>/dev/null || true)"
        [ "$s" = "$EVAL_SEED" ] && echo "$f"
      done | xargs -r ls -t 2>/dev/null | head -1 || true)"
  fi
  [ -n "$eval_src" ] && [ -f "$eval_src" ] || {
    echo "ERROR: no eval JSON with seed $EVAL_SEED under $LOCAL_OUTPUTS/$RUN* (set EVAL_JSON=<path>)."
    echo "       Run the final-checkpoint eval first: uv run python -m eval_speedrun --run $RUN-final-eval \\"
    echo "           --eval-checkpoint $run_dir/step_<N> --eval-seed $EVAL_SEED ..."
    exit 1
  }
  echo ">> eval artifact: $eval_src"
  eval_json="$(basename "$eval_src")"
  cp "$eval_src" "$stage/$eval_json"
  eval_roll_src="${eval_src%.json}.rollouts.jsonl.gz"
  if [ -f "$eval_roll_src" ]; then
    eval_roll="$(basename "$eval_roll_src")"
    cp "$eval_roll_src" "$stage/$eval_roll"
  else
    echo ">> WARN: no rollouts beside the eval JSON — verify_record will skip re-scoring"
  fi
  cp "$log_path" "$stage/$log_name"
  [ -f "$run_dir/metrics.jsonl" ] && cp "$run_dir/metrics.jsonl" "$stage/metrics.jsonl"
  if [ -z "$VERIFY_OF" ] && [ -f "$run_dir/final_rollouts.jsonl.gz" ]; then
    cp "$run_dir/final_rollouts.jsonl.gz" "$stage/final_rollouts.jsonl.gz"
  fi
  mkdir -p "$stage/source"
  if [ -f "$run_dir/source/speedrun.py" ]; then
    cp "$run_dir/source/speedrun.py" "$stage/source/speedrun.py"
    echo ">> copied source snapshot"
  else
    echo ">> source snapshot not found in run dir; report generation will backfill from train log"
  fi
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
rm -f "$DEST/metrics_seed${TRAIN_SEED}.jsonl"
[ -f "$stage/metrics.jsonl" ] && \
  cp "$stage/metrics.jsonl" "$DEST/metrics_seed${TRAIN_SEED}.jsonl"
[ -n "$eval_roll" ] && [ -f "$stage/$eval_roll" ] && \
  cp "$stage/$eval_roll" "$DEST/eval_seed${TRAIN_SEED}.rollouts.jsonl.gz"
[ -z "$VERIFY_OF" ] && [ -f "$stage/final_rollouts.jsonl.gz" ] && \
  cp "$stage/final_rollouts.jsonl.gz" "$DEST/final_rollouts_seed${TRAIN_SEED}.jsonl.gz"
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
fi

# A submission adds its README leaderboard row; verification refreshes that same row. Both redraw
# the figures so the rolling training plot gains its matched two-seed band after the rerun.
report_args=("$RECORD" --update-leaderboard)
echo ">> generating report + plots for $RECORD"
uv run python ../make_record_report.py "${report_args[@]}"
echo ">> verifying $RECORD (submission + verification)"
uv run python verify_record.py "$RECORD"
echo
if [ -n "$VERIFY_OF" ]; then
  echo ">> DONE. Verification appended to $RECORD; its README now has a ## Verification section."
else
  echo ">> DONE. Leaderboard row added + leaderboard/training figures redrawn. Next:"
  echo "   1) fill in the placeholder '## Idea' section in $DEST/README.md"
  echo "   2) fill in the new row's Description + Contributors in README.md"
  echo "   3) run a verification rerun (different seed): RUN=<VRUN> VERIFY_OF=$DEST ./assemble_record.sh"
  echo "   4) remove any superseded record dir, then open the PR"
fi
