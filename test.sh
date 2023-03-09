set -ex
export CUDA_DEVICE_ORDER='PCI_BUS_ID'
export CUDA_VISIBLE_DEVICES="$(seq -s ',' 0 $(($(nvidia-smi -L | wc -l) - 1)))"

python3 main.py \
  --env BreakoutNoFrameskip-v4 \
  --case atari \
  --opr test \
  --seed 0 \
  --num_gpus 1 \
  --num_cpus 20 \  
  --force \
  --test_episodes 32 \
  --load_model \
  --amp_type 'torch_amp' \
  --model_path 'model.p' \
  --info 'Test'