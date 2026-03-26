#!/bin/bash
# set -euo pipefail

# Run training+eval experiments on remote machines via execute_remote.sh.
# Parallel across machines: each host runs at most one job. As soon as a machine
# is free, the next pending config is assigned (fill order follows the machine
# list below). Same machine list as execute_entities.sh.
#
# Usage:
#   ./execute_experiments.sh                    # all configs/novqa_*.yaml and configs/withvqa_*.yaml (sorted)
#   ./execute_experiments.sh configs/foo.yaml   # explicit config(s)
#
# Eval JSON: ${EVAL_HOME:-$HOME}/eval_<experiment_name>.json on the worker (set EVAL_HOME if
# the remote home path differs from this machine's $HOME).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
EVAL_BASE="${EVAL_HOME:-$HOME}"

############################################
# Machine list (keep in sync with execute_entities.sh)
############################################
MACHINES=(
    ain ardennes carmor charente cher creuse
    dordogne doubs essonne finistere gironde indre
    jura landes loire manche marne mayenne morbihan
    moselle saone somme vendee vosges
)

############################################
# Config list
############################################
if [[ $# -gt 0 ]]; then
    CONFIGS=("$@")
else
    mapfile -t CONFIGS < <(ls -1 configs/novqa_*.yaml configs/withvqa_*.yaml 2>/dev/null | sort || true)
fi

if [[ ${#CONFIGS[@]} -eq 0 ]]; then
    echo "No configs to run. Pass yaml paths as arguments or ensure configs/novqa_*.yaml and configs/withvqa_*.yaml exist."
    exit 1
fi

############################################
# Remote done file (same convention as execute_remote.sh, on the worker host)
############################################
remote_done_exists() {
    local machine=$1
    ssh -o BatchMode=yes -o ConnectTimeout=30 "$machine" "test -f \"\$HOME/gen_ai/${machine}.done\""
}

clear_remote_done() {
    local machine=$1
    ssh -o BatchMode=yes -o ConnectTimeout=30 "$machine" "rm -f \"\$HOME/gen_ai/${machine}.done\""
}

# Wait until any machine in BUSY has ~/gen_ai/<machine>.done; remove that key from BUSY.
# BUSY is an associative array: machine name -> config path (for logging).
wait_for_any_busy_machine() {
    declare -n _busy_ref=$1
    local machine
    echo "[WAIT] Waiting for any of: ${!_busy_ref[*]} (remote: ~/gen_ai/<machine>.done) ..."
    while true; do
        for machine in "${!_busy_ref[@]}"; do
            if remote_done_exists "$machine"; then
                echo "[DONE] ${machine} finished $(basename "${_busy_ref[$machine]}" .yaml)."
                unset "_busy_ref[$machine]"
                return 0
            fi
        done
        sleep 30
    done
}

extract_model_from_yaml() {
    local f=$1
    local line
    line=$(grep -E '^model:' "$f" | head -1 || true)
    line=${line#model:}
    line=${line//\"/}
    line=${line//\'/}
    line=$(echo "$line" | sed -e 's/^[[:space:]]*//;s/[[:space:]]*$//')
    echo "${line:-Qwen/Qwen3.5-0.8B-Base}"
}

REMOTE_SCRIPT="${SCRIPT_DIR}/execute_remote.sh"

declare -A BUSY # machine -> config path (active job)
next_idx=0
run_i=0
PICKED_CONFIG=""

# Sets PICKED_CONFIG; must not run in a subshell (next_idx must update here).
pick_next_config() {
    while (( next_idx < ${#CONFIGS[@]} )); do
        local c=${CONFIGS[next_idx]}
        ((next_idx++))
        if [[ -f "$c" ]]; then
            PICKED_CONFIG=$c
            return 0
        fi
        echo "[SKIP] Not a file: $c" >&2
    done
    return 1
}

launch_on_machine() {
    local machine=$1
    local config=$2
    local exp_name model cmd

    exp_name=$(basename "$config" .yaml)
    model=$(extract_model_from_yaml "$config")

    clear_remote_done "$machine"

    # execute_remote.sh wraps the remote command in single quotes, so $HOME inside cmd would
    # not expand on the worker; use an absolute path (typical when $HOME is shared across nodes).
    # Avoid single quotes inside cmd (they would break the remote bash -c '...' wrapper).
    cmd="uv sync && uv run python -u src/train_with_augmentation.py -c ${config} && uv run python -u src/evaluate.py --model ${model} --checkpoint results/${exp_name}/final --output ${EVAL_BASE}/eval_${exp_name}.json"

    ((run_i++))
    echo
    echo "[RUN ${run_i}/${#CONFIGS[@]}] ${machine} ← ${config}"
    bash "$REMOTE_SCRIPT" "$machine" "$cmd"
    BUSY[$machine]=$config
}

while (( next_idx < ${#CONFIGS[@]} )) || (( ${#BUSY[@]} > 0 )); do
    # Assign every idle machine until no pending configs or no free hosts.
    for machine in "${MACHINES[@]}"; do
        (( next_idx < ${#CONFIGS[@]} )) || break
        [[ -n "${BUSY[$machine]:-}" ]] && continue
        pick_next_config || break
        launch_on_machine "$machine" "$PICKED_CONFIG"
    done

    (( ${#BUSY[@]} > 0 )) || break
    wait_for_any_busy_machine BUSY
done

echo
echo "All experiments finished."
