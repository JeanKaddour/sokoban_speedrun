#!/usr/bin/env bash
# Assemble a leaderboard record — or its verification rerun — from a finished Modal run.
#
# Both modes need the train run AND its final-checkpoint eval finished + committed to the volume
# (eval reuses RUN_NAME so its JSON lands in the same /outputs/<RUN>/ dir):
#   MAX_STEPS=<steps> RUN_NAME=<RUN> uv run modal run --detach modal_app.py
#   EVAL_CHECKPOINT=/vol/outputs/<RUN>/step_<FINAL_STEP> RUN_NAME=<RUN> uv run modal run modal_app.py
#
# SUBMISSION (the record itself):
#   RUN=<RUN> DEST=records/<date>_01_<name> IDEA_FILE=../reports/baseline-rerun-idea.md ./assemble_record.sh
#
# VERIFICATION (an independent rerun, dropped into the record's verification/ subdir — run a SECOND
# training+eval with a different seed first, then):
#   RUN=<VRUN> TRAIN_SEED=<vseed> VERIFY_OF=records/<date>_01_<name> \
#       VERIFIER="@maintainer  PR#12" ./assemble_record.sh
#
# Pulls only the record artifacts (not the multi-GB checkpoint), including the source snapshot when
# present, renames to the seed convention, then regenerates the record report + verifies
# (verify_record checks submission AND verification).
set -euo pipefail

RUN="${RUN:?set RUN to the Modal run name (outputs/<RUN>/ on the volume)}"
TRAIN_SEED="${TRAIN_SEED:-42}"   # labels the record files; the documented command's default seed
EVAL_SEED="${EVAL_SEED:-12345}"  # pinned eval sampling seed (the JSON's sampling.seed)
STEP="${STEP:-}"                 # optional eval checkpoint step; inferred from eval artifacts if unset
VOL="${VOL:-nanochat-rl-hf}"
IDEA_FILE="${IDEA_FILE:-}"
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

mkdir -p "$DEST"
cp "$stage/$log_name"  "$DEST/train_log_seed${TRAIN_SEED}.txt"
cp "$stage/$eval_json" "$DEST/eval_seed${TRAIN_SEED}.json"
cp "$stage/$eval_roll" "$DEST/eval_seed${TRAIN_SEED}.rollouts.jsonl.gz"
[ -z "$VERIFY_OF" ] && cp "$stage/final_rollouts.jsonl.gz" "$DEST/final_rollouts_seed${TRAIN_SEED}.jsonl.gz"
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

echo ">> generating report + plots for $RECORD"
uv run python ../make_record_report.py "$RECORD"
echo ">> verifying $RECORD (submission + verification)"
uv run python verify_record.py "$RECORD"
echo ">> regenerating LLM track hero (records/hero.gif)"
uv run --with pillow python records/plot_hero_animation.py "$RECORD" --out records/hero.gif
echo
if [ -n "$VERIFY_OF" ]; then
  echo ">> DONE. Verification appended to $RECORD; its README now has a ## Verification section."
else
  echo ">> DONE. Next: run a verification rerun (VERIFY_OF=$DEST ...), then update the README"
  echo "   leaderboard row (date, record time, pass@1/CI) and remove any superseded record dir."
  echo "   Finally refresh the leaderboard figures: uv run python ../make_record_report.py --leaderboard"
fi
