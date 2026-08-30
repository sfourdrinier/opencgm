#!/usr/bin/env bash
# Keep a set of pretraining runs alive across host crashes, OOM kills and driver resets.
#
# Every run is checkpointed to ckpt_last.pt each epoch and its batch order is a pure function of
# (seed, epoch), so a resume replays the step that would have run had nothing been interrupted.
# That makes automatic restart safe in a way it usually is not: this loses at most one epoch and
# does not silently change what the run is.
#
# The one thing that would make it unsafe is a resume into different architecture flags, which is
# exactly what `check_architecture` refuses. So a run whose flags no longer match its checkpoint
# fails loudly and stays down rather than quietly becoming a different experiment. That is the
# intended behaviour: a wedged run is recoverable, a silently mongrel run is not.
#
#   ./scripts/watchdog.sh runs.txt
#
# where runs.txt holds one launch command per line, blank lines and # comments ignored. Each
# command must be idempotent: it needs --resume pointing at its own ckpt_last.pt, and it must
# tolerate that file not existing yet (train() falls through to a fresh start).

set -uo pipefail
cd "$(dirname "$0")/.."

SPEC=${1:?usage: watchdog.sh runs.txt}
INTERVAL=${INTERVAL:-120}
LOGDIR=logs
mkdir -p "$LOGDIR"

# A run is identified by its --out directory, which is unique per run and appears verbatim in the
# process command line. Matching on that rather than on the tag avoids the trap where the pattern
# also matches this script or the ssh command that launched it.
out_dir_of() { sed -n 's/.*--out \([^ ]*\).*/\1/p' <<<"$1"; }

alive() {
  local out=$1 pid comm
  for pid in $(pgrep -f -- "--out $out " 2>/dev/null); do
    comm=$(cat "/proc/$pid/comm" 2>/dev/null) || continue
    case "$comm" in python*|uv) return 0 ;;
  esac
  done
  return 1
}

declare -A FAILURES

while :; do
  while IFS= read -r cmd; do
    [[ -z "$cmd" || "$cmd" == \#* ]] && continue
    out=$(out_dir_of "$cmd")
    [[ -z "$out" ]] && { echo "$(date -Is) SKIP no --out in: $cmd"; continue; }
    name=$(basename "$out")

    if alive "$out"; then continue; fi

    # A run that has reached its last epoch is finished, not crashed. `ckpt_final.pt` is written
    # only on normal completion, so its presence is the terminal condition.
    if [[ -f "$out/ckpt_final.pt" ]]; then continue; fi

    fails=${FAILURES[$name]:-0}
    if (( fails >= 5 )); then continue; fi

    # Five consecutive restarts that never survive to the next check means something structural --
    # a refused resume, a missing cache, a full disk. Backing off stops the watchdog from turning
    # one broken run into a thousand log lines.
    FAILURES[$name]=$(( fails + 1 ))
    echo "$(date -Is) RESTART $name (attempt $(( fails + 1 )))" | tee -a "$LOGDIR/watchdog.log"
    setsid nohup bash -c "$cmd" >>"$LOGDIR/$name.log" 2>&1 </dev/null &
    sleep 20
    if alive "$out"; then
      FAILURES[$name]=0
    else
      echo "$(date -Is) FAILED-TO-START $name, see $LOGDIR/$name.log" | tee -a "$LOGDIR/watchdog.log"
    fi
  done < "$SPEC"
  sleep "$INTERVAL"
done
