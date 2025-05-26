#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1

# PYTHONPATH에 프로젝트 루트 추가 (스크립트 위치 기준 상위 2단계)
export PYTHONPATH="$(cd "$(dirname "$0")/../.." && pwd)":$PYTHONPATH

DATA_ROOT="/data2"  # 필요시 /data2 등으로 변경

# 첫 번째 인자로 use_layernorm 값을 받음 (기본값 true)
use_layernorm=true  # true 또는 false로 직접 지정

# learning rate도 직접 지정
lr=1e-3

n_gpu=2
dist_port=12400
template="options/train/NAFNetV3_RGB/template_rgb_enc2.j2"
num_worker_per_gpu=2
batch_size_per_gpu=8
shuffle1=1
end_type=0
mid_type=0
epoch=80
sca=true
ch_list=(64)
ch_mid_list=(64)
mid_list=(3)
length=${#ch_list[@]}
n_mids=${#mid_list[@]}

TRAIN_SET_SIZE=30380

auto_total_iter() {
  local epoch=$1
  local batch_size=$2
  local n_gpu=$3
  local train_set_size=$4
  local num_iter_per_epoch=$(( (train_set_size + batch_size * n_gpu - 1) / (batch_size * n_gpu) ))
  echo $(( epoch * num_iter_per_epoch ))
}

for ((i=0; i<length; i++)); do
	for ((j=0; j<n_mids; j++)); do
    total_iter=$(auto_total_iter $epoch $batch_size_per_gpu $n_gpu $TRAIN_SET_SIZE)
    jinja2 "$template" \
      -D data_root="$DATA_ROOT" \
      -D shuffle1=${shuffle1} \
      -D mid=${mid_list[$j]} -D width=${ch_list[$i]} -D ch_mid=${ch_mid_list[$i]} \
      -D end_type=${end_type} -D mid_type=${mid_type} \
      -D total_iter=$total_iter \
      -D _num_gpu=$n_gpu -D _num_worker_per_gpu=$num_worker_per_gpu -D _batch_size_per_gpu=$batch_size_per_gpu \
      -D sca=$sca \
      -D use_layernorm=$use_layernorm \
      -D lr=$lr \
      > temp_config.yml
		torchrun --nproc_per_node=${n_gpu} --nnodes=1 --rdzv-backend=c10d \
		--rdzv-endpoint=localhost:${dist_port} \
		basicsr/train_param.py --launcher pytorch \
		--template temp_config.yml
		rm temp_config.yml
	done
done
