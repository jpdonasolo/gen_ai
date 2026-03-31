#!/bin/bash
set -e

############################################
# CONFIGURATION
############################################
TOTAL_DOCS=621
DOCS_PER_MACHINE=20

BASE_DIR="$HOME/gen_ai"
WORKDIR="/Data/joao.giordani-donasolo"
IMAGES_DIR="output/textbook_of_pathology"
LOG_DIR="$BASE_DIR/logs"

MACHINES=(
    ain ardennes carmor charente creuse
    dordogne doubs essonne finistere gironde indre
    jura landes loire manche marne mayenne morbihan
    moselle saone somme vendee vosges
)

############################################
# SYNC
############################################
echo "[SYNC] Syncing current folder to $BASE_DIR ..."
rsync -a \
    --exclude='huggingface' \
    --exclude='.venv' \
    --exclude='.git' \
    --exclude='data' \
    --exclude='results' \
    --exclude='wandb' \
    . ~/gen_ai

############################################
# INIT
############################################
mkdir -p "$LOG_DIR"

N_JOBS=$(( (TOTAL_DOCS + DOCS_PER_MACHINE - 1) / DOCS_PER_MACHINE ))
N_MACHINES=${#MACHINES[@]}

echo "[INFO] $TOTAL_DOCS docs → $N_JOBS jobs, $DOCS_PER_MACHINE docs each, up to $N_MACHINES machines in parallel."

############################################
# FUNCTIONS
############################################
run_remote() {
    local machine=$1
    local start=$2
    local end=$3
    local done_file=$4
    local log=$5

    local cmd="
        rsync -a ~/gen_ai $WORKDIR;
        cd $WORKDIR/gen_ai;
        uv sync;
        uv run python -u src/extract_entities.py \
            --images-dir $IMAGES_DIR \
            --start $start \
            --end $end \
            --relations-out-dir $IMAGES_DIR/relations_${start}_${end}.jsonl \
            --restart-relations;
        cp $IMAGES_DIR/relations_${start}_${end}.jsonl $BASE_DIR/$IMAGES_DIR/relations_${start}_${end}.jsonl
    "
    echo "[LAUNCH] $machine → $cmd"

    ssh "$machine" bash << EOF
nohup bash -c '
    $cmd
    touch $done_file
' > $log 2>&1 < /dev/null &
EOF
}

# Returns index of a free machine (one whose slot has no pending done_file),
# or blocks until one becomes available.
wait_for_free_machine() {
    while true; do
        for idx in $(seq 0 $((N_MACHINES - 1))); do
            if [ -z "${MACHINE_DONE[$idx]}" ] || [ -f "${MACHINE_DONE[$idx]}" ]; then
                echo $idx
                return
            fi
        done
        sleep 5
    done
}

############################################
# LAUNCH — job queue
############################################
declare -A MACHINE_DONE   # machine_idx → current done_file (empty = free)
ALL_DONE_FILES=()

for job in $(seq 0 $((N_JOBS - 1))); do
    start=$((job * DOCS_PER_MACHINE))
    end=$((start + DOCS_PER_MACHINE))
    if [ "$end" -gt "$TOTAL_DOCS" ]; then end=$TOTAL_DOCS; fi

    machine_idx=$(wait_for_free_machine)
    machine=${MACHINES[$machine_idx]}

    done_file="$LOG_DIR/job_${job}_${start}_${end}.done"
    log="$LOG_DIR/job_${job}_${start}_${end}.log"
    rm -f "$done_file"

    run_remote "$machine" "$start" "$end" "$done_file" "$log"

    MACHINE_DONE[$machine_idx]="$done_file"
    ALL_DONE_FILES+=("$done_file")
done

############################################
# WAIT — all remaining jobs
############################################
echo
echo "[WAIT] Waiting for all jobs to finish..."
for done_file in "${ALL_DONE_FILES[@]}"; do
    while [ ! -f "$done_file" ]; do sleep 30; done
    echo "[DONE] $done_file"
done

echo
echo "All jobs finished."