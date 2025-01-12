#!/bin/bash

# --------------------------------------------
# Script to Run Multiple Training Jobs in Parallel
# --------------------------------------------

# Function to run a group of commands sequentially
run_group() {
    group_name="$1"
    shift
    commands=("$@")
    echo "Starting Group: $group_name"
    for cmd in "${commands[@]}"; do
        echo "[$group_name] Running: $cmd"
        eval "$cmd"
        if [ $? -ne 0 ]; then
            echo "[$group_name] Command failed: $cmd"
            exit 1
        fi
    done
    echo "Completed Group: $group_name"
}

# Define commands for each model size group

# Group 1: num_layers=1, hidden_dim=64, num_heads=1
group1_commands=(
    "python char_scaling_laws.py --experiment_name ts_d64_l1_h1_t1e6 --num_train_tokens 1000000 --num_layers 1 --hidden_dim 64 --num_heads 1"
    "python char_scaling_laws.py --experiment_name ts_d64_l1_h1_t2e6 --num_train_tokens 2000000 --num_layers 1 --hidden_dim 64 --num_heads 1"
    "python char_scaling_laws.py --experiment_name ts_d64_l1_h1_t4e6 --num_train_tokens 4000000 --num_layers 1 --hidden_dim 64 --num_heads 1"
    "python char_scaling_laws.py --experiment_name ts_d64_l1_h1_t8e6 --num_train_tokens 8000000 --num_layers 1 --hidden_dim 64 --num_heads 1"
    "python char_scaling_laws.py --experiment_name ts_d64_l1_h1_t1.6e7 --num_train_tokens 160000000 --num_layers 1 --hidden_dim 64 --num_heads 1"
)

# Group 2: num_layers=2, hidden_dim=128, num_heads=2
group2_commands=(
    "python char_scaling_laws.py --experiment_name ts_d128_l2_h2_t1e6 --num_train_tokens 1000000 --num_layers 2 --hidden_dim 128 --num_heads 2"
    "python char_scaling_laws.py --experiment_name ts_d128_l2_h2_t2e6 --num_train_tokens 2000000 --num_layers 2 --hidden_dim 128 --num_heads 2"
    "python char_scaling_laws.py --experiment_name ts_d128_l2_h2_t4e6 --num_train_tokens 4000000 --num_layers 2 --hidden_dim 128 --num_heads 2"
    "python char_scaling_laws.py --experiment_name ts_d128_l2_h2_t8e6 --num_train_tokens 8000000 --num_layers 2 --hidden_dim 128 --num_heads 2"
    "python char_scaling_laws.py --experiment_name ts_d128_l2_h2_t1.6e7 --num_train_tokens 160000000 --num_layers 2 --hidden_dim 128 --num_heads 2"
)

# Group 3: num_layers=4, hidden_dim=256, num_heads=4
group3_commands=(
    "python char_scaling_laws.py --experiment_name ts_d256_l4_h4_t1e6 --num_train_tokens 1000000 --num_layers 4 --hidden_dim 256 --num_heads 4"
    "python char_scaling_laws.py --experiment_name ts_d256_l4_h4_t2e6 --num_train_tokens 2000000 --num_layers 4 --hidden_dim 256 --num_heads 4"
    "python char_scaling_laws.py --experiment_name ts_d256_l4_h4_t4e6 --num_train_tokens 4000000 --num_layers 4 --hidden_dim 256 --num_heads 4"
    "python char_scaling_laws.py --experiment_name ts_d256_l4_h4_t8e6 --num_train_tokens 8000000 --num_layers 4 --hidden_dim 256 --num_heads 4"
    "python char_scaling_laws.py --experiment_name ts_d256_l4_h4_t1.6e7 --num_train_tokens 160000000 --num_layers 4 --hidden_dim 256 --num_heads 4"
)

# Launch each group in the background
run_group "Group1_d64_l1_h1" "${group1_commands[@]}" &
pid1=$!

run_group "Group2_d128_l2_h2" "${group2_commands[@]}" &
pid2=$!

run_group "Group3_d256_l4_h4" "${group3_commands[@]}" &
pid3=$!

# Wait for all background jobs to finish
wait $pid1 $pid2 $pid3

echo "All training jobs have completed."