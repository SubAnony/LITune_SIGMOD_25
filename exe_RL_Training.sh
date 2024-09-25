if [ -z "$3" ]; then
    seed_value=0
else
    seed_value=$3
fi

if [ -z "$4" ]; then
    file_name=data_11
else
    file_name=$4
fi


echo "Start Training......"

echo "RL Training"

# Simple Episodic Training

# Default RL model is Vanilla DDPG, you can replace it by DDPG_Context, TD3
python3 ./scripts/RL_controller_offline.py --RL_policy DDPG --load_model default --save_model True --Index $2 --query_type $1

# Supporting Meta-RL Training
# python3 ./scripts/meta_RL_offline.py --RL_policy DDPG --load_model default --save_model True --Index $2 --query_type $1

#!/bin/bash
# Uncomment the following lines for Simplified Meta RL training pipeline
# EPOCHS=100
# VALIDATION_INTERVAL=10

# for ((i=1; i<=EPOCHS; i++)); do
#     # Step 1: Initialize the DRL agent (only for the first epoch)
#     if [ "$i" -eq 1 ]; then
#         python3 ./scripts/RL_controller_offline.py --RL_policy DDPG --load_model default --save_model True --Index $2 --query_type $1 --data_file data_0 --mode Initialization
#     fi

#     # Step 2: Update model parameters
#     python3 ./scripts/RL_controller_offline.py --RL_policy DDPG --load_model default --save_model True --Index $2 --query_type $1 --data_file data_10 --mode training

#     # Step 3: Periodically evaluate on the validation set
#     if [ $(($i % $VALIDATION_INTERVAL)) -eq 0 ]; then

#         validation_output=$(python3 ./scripts/RL_controller_offline.py --RL_policy DDPG --load_model default --save_model True --Index $2 --query_type $1 --data_file data_20 --mode validate | grep "VALIDATION_SCORE:")
#         validation_score=${validation_output#*:}  # This removes "VALIDATION_SCORE:" and keeps only the score

#         echo "Epoch $i: Validation Score = $validation_score"

#         # Here, you can include logic to adjust training based on validation_score
#     fi

#     # Step 4: Repeat until convergence (handled by loop)
# done



