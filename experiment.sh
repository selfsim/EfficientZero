set -ex

python3 main.py \
  --env AlienNoFrameskip-v0 \
  --case atari \
  --opr train \
  --force \
  --num_gpus 4 \
  --num_cpus 16 \
  --cpu_actor 16 \
  --gpu_actor 8 \
  --seed 0 \
  --p_mcts_num 4 \
  --use_priority \
  --use_max_priority \
  --amp_type 'torch_amp' \
  --info 'E0 - Alien_test'