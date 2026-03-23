rsync -a --exclude='huggingface' --exclude='.venv' --exclude='data' . ~/gen_ai
set -e

############################################
# CONFIGURATION
############################################
BASE_DIR="$HOME/gen_ai"
LOG_DIR="$BASE_DIR/logs"

MACHINES=(
    ain allier ardennes carmor charente cher creuse 
    dordogne doubs essonne finistere gironde indre 
    jura landes loire manche marne mayenne morbihan 
    moselle saone somme vendee vosges 
)
############################################
# INIT
############################################
mkdir -p "$LOG_DIR"
N_MACHINES=${#MACHINES[@]}
JOB_ID=0

############################################
# FUNCTIONS
############################################
run_remote() {
    local machine=$1
    local cmd=$2
    local done_file=$3
    local log=$4

    echo "[LAUNCH] $machine → $cmd"

    ssh "$machine" "
        cd $BASE_DIR &&
        nohup bash -c '$cmd; touch $done_file' > $log 2>&1 &
    "
}
############################################
# MAIN LOOP
############################################
for cycle in $(seq 1 $N_CYCLES); do
    echo
    echo "=========================================="
    echo "        CYCLE $cycle / $N_CYCLES"
    echo "=========================================="

    ########################################
    # PHASE A — poisoner init (PARALLEL)
    ########################################
    echo "[PHASE A] Initial poisoner runs"
    DONE_FILES=()
    JOB_CMDS=()

    # Préparer tous les jobs
    for aggregator in "${AGGREGATOR[@]}"; do
        JOB_CMDS+=("python run_experiment.py federated_experiments/${NUM_POISONED}vs${NUM_CLEAN}/${DATASET}/${ATTACK}/${aggregator}/gen_labels")
    done

    JOB_ID=0
    for cmd in "${JOB_CMDS[@]}"; do
        machine=${MACHINES[$((JOB_ID % N_MACHINES))]}

        safe_cmd=${cmd//[ \/]/_}

        done_file="$LOG_DIR/cycle${cycle}_init_${safe_cmd}.done"
        log="$LOG_DIR/cycle${cycle}_init_${safe_cmd}.log"
        rm -f "$done_file"

        run_remote "$machine" "$cmd" "$done_file" "$log" &

        DONE_FILES+=("$done_file")
        JOB_ID=$((JOB_ID + 1))
    done

done
