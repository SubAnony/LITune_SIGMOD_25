echo "-----------------------------------------------------------------------------------------"

echo "Start Testing......"

echo "RL_stream_Controller"

#Simple O2 system design

# Default RL model is Vanilla DDPG, you can replace it by DDPG_Context, TD3
python3 ./scripts/RL_stream_controller.py --RL_policy DDPG --data_file $file_name --load_model default --Index $2 --seed $seed_value --query_type $1

#Full O2 system design
# python3 ./scripts/RL_O2.py --RL_policy DDPG --data_file $file_name --load_model default --Index $2 --seed $seed_value --query_type $1


echo "End Testing......"