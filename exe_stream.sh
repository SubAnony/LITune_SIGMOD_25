echo "Start Testing Streaming Data......"

echo "RL eval"
# Default RL model is Vanilla DDPG, you can replace it by DDPG_Context, TD3
python3 ./scripts/RL_eval.py --RL_policy DDPG --Index $2 --load_model default --seed 0 --query_type $1

wait

echo "Default Setting"

i=0
for file in $(ls ./data_SOSD | sort -V); do
    while [ $i -ge 11 ] && [ $i -le 21 ]; do
        echo "tuning on $i-th file: $file"
		file_name=`basename $file .txt` 
		python3 ./scripts/data_stream_control.py --data_file $file_name --search_method default --Index $2 --seed 0 --query_type $1
        break
    done
    i=$((i+1))
done


wait


echo "Random Search"

i=0
for file in $(ls ./data_SOSD | sort -V); do
    while [ $i -ge 11 ] && [ $i -le 21 ]; do
        echo "tuning on $i-th file: $file"
		file_name=`basename $file .txt`
		python3 ./scripts/data_stream_control.py --data_file $file_name --search_method random_search --Index $2 --seed 0 --query_type $1
        break
    done
    i=$((i+1))
done

wait


echo "Heuristic Search"

i=0
for file in $(ls ./data_SOSD | sort -V); do
    while [ $i -ge 11 ] && [ $i -le 21 ]; do
        echo "tuning on $i-th file: $file"
		file_name=`basename $file .txt`
		python3 ./scripts/data_stream_control.py --data_file $file_name --search_method heuristic_search --Index $2 --seed 0 --query_type $1
        break
    done
    i=$((i+1))
done

wait


echo "BO"


i=0
for file in $(ls ./data_SOSD | sort -V); do
    while [ $i -ge 11 ] && [ $i -le 21 ]; do
        echo "tuning on $i-th file: $file"
		file_name=`basename $file .txt`
		python3 ./scripts/data_stream_control.py --data_file $file_name --search_method BO --Index $2 --seed 0 --query_type $1
        # python3 ./data_stream_control.py --data_file $file_name --search_method BO --Index $2 --seed 1 --query_type $1
        # python3 ./data_stream_control.py --data_file $file_name --search_method BO --Index $2 --seed 2 --query_type $1
        break
    done
    i=$((i+1))
done

wait

echo "grid_search"

i=0
for file in $(ls ./data_SOSD | sort -V); do
    while [ $i -ge 11 ] && [ $i -le 21 ]; do
        echo "tuning on $i-th file: $file"
		file_name=`basename $file .txt`
		python3 ./scripts/data_stream_control.py --data_file $file_name --search_method grid_search --Index $2 --seed 0 --query_type $1
        break
    done
    i=$((i+1))
done

wait


echo "End Testing......"
