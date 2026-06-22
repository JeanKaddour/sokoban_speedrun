#!/usr/bin/env bash
# Assemble a non-LLM leaderboard record — or its verification rerun — from a finished Modal run.
#
# A single train call also runs the final held-out eval (modal_app_non_llm.train does train+eval in
# one shot), so both the run log and the eval JSON land in /outputs/<RUN>/ on the volume:
#   RUN_NAME=<RUN> DIFFICULTY=4 TOTAL_TIMESTEPS=<steps> uv run modal run --detach modal_app_non_llm.py
# (Re-eval an existing checkpoint into the same dir with EVAL_CHECKPOINT=/vol/outputs/<RUN>/final.pt
#  RUN_NAME=<RUN> ... — only needed when you didn't eval during training.)
#
# SUBMISSION (the record itself):
#   RUN=<RUN> DEST=records/<date>_01_<name> IDEA_FILE=../reports/non-llm-idea.md ./assemble_record.sh
#
# VERIFICATION (an independent rerun, dropped into the record's verification/ subdir — run a SECOND
# train+eval with a different seed first via EXTRA_ARGS="--seed <vseed>", then):
#   RUN=<VRUN> TRAIN_SEED=<vseed> VERIFY_OF=records/<date>_01_<name> \
#       VERIFIER="@maintainer  PR#12" ./assemble_record.sh
#
# Pulls only the record artifacts (not the multi-MB checkpoint): the run log, eval JSON, and source
# snapshot, renamed to the seed convention, then regenerates the record report + verifies
# (verify_record checks submission AND verification).
set -euo pipefail

RUN="${RUN:?set RUN to the Modal run name (outputs/<RUN>/ on the volume)}"
TRAIN_SEED="${TRAIN_SEED:-42}"   # labels the record files; the documented command's default seed
EVAL_SEED="${EVAL_SEED:-12345}"  # pinned eval seed (the JSON's sampling.seed / eval_step<...>_seed<EVAL_SEED>)
STEP="${STEP:-}"                 # optional eval checkpoint step; inferred from eval artifacts if unset
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
echo ">> eval artifact: $eval_json"

echo ">> pulling record artifacts (not the checkpoint)"
for f in "$log_name" "$eval_json"; do
  modal volume get "$VOL" "/outputs/$RUN/$f" "$stage/" --force
done
mkdir -p "$stage/source"
if modal volume get "$VOL" "/outputs/$RUN/source/speedrun_non_llm.py" "$stage/source/" --force; then
  echo ">> pulled source snapshot"
else
  echo ">> source snapshot not found in run output; report generation will backfill from train log"
fi

mkdir -p "$DEST"
cp "$stage/$log_name"  "$DEST/train_log_seed${TRAIN_SEED}.txt"
cp "$stage/$eval_json" "$DEST/eval_seed${TRAIN_SEED}.json"
if [ -f "$stage/source/speedrun_non_llm.py" ]; then
  mkdir -p "$DEST/source"
  cp "$stage/source/speedrun_non_llm.py" "$DEST/source/speedrun_seed${TRAIN_SEED}.py"
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
uv run python ../make_record_report.py "$RECORD" --track non-llm
echo ">> verifying $RECORD (submission + verification)"
uv run python verify_record.py "$RECORD"
echo
if [ -n "$VERIFY_OF" ]; then
  echo ">> DONE. Verification appended to $RECORD; its README now has a ## Verification section."
else
  echo ">> DONE. Next: run a verification rerun (VERIFY_OF=$DEST TRAIN_SEED=<vseed> ...), then update"
  echo "   the README leaderboard row (date, record time, pass@1/CI) and remove any superseded dir."
fi
