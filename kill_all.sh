#!/bin/bash

MACHINES=(
ain ardennes carmor charente cher creuse
dordogne doubs essonne finistere gironde indre
jura landes loire manche marne mayenne morbihan
moselle saone somme vendee vosges
)

for machine in "${MACHINES[@]}"; do
    echo "[KILL] $machine..."
    ssh "$machine" "pkill -u $USER -f 'python|uv run'" && echo "[OK] $machine" || echo "[SKIP] $machine (nothing or unreachable)"
done