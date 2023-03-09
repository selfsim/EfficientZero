set -ex
#export CUDA_DEVICE_ORDER='PCI_BUS_ID'
#export CUDA_VISIBLE_DEVICES=0,1,2,3

python3 main.py \
  --env alien \
  --case atari \
  --opr train \
  --force \
  --num_gpus 8 \
  --num_cpus 32 \
  --cpu_actor 14 \
  --gpu_actor 20 \
  --seed 0 \
  --p_mcts_num 4 \
  --use_priority \
  --use_max_priority \
  --amp_type 'torch_amp' \
  --info 'E0 - Alien_test' \
  --save_video
