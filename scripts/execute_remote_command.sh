
machine=$1
cmd=$2

# rsync
rsync -a \
    --exclude='huggingface' \
    --exclude='.venv' \
    --exclude='.git' \
    --exclude='data' \
    --exclude='results' \
    --exclude='wandb' \
    --exclude='*.json' \
    . ~/gen_ai


# ssh to target machine
# clean data folder
# rsync with home folder
# execute command
done_file="~/gen_ai/${machine}.done"
log_file="~/gen_ai/${machine}.log"

ssh "$machine" bash << EOF
nohup bash -c '
    rsync -a ~/gen_ai /Data/joao.giordani-donasolo;
    cd /Data/joao.giordani-donasolo/gen_ai;
    $cmd;
    touch $done_file
' > $log_file 2>&1 < /dev/null &
EOF
