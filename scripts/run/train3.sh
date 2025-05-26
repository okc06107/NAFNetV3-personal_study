export CUDA_VISIBLE_DEVICES=6,7

n_gpu=2
dist_port=12358
template="options/train/NAFNetV3/template_baseline_enc3.j2"
num_worker_per_gpu=2
batch_size_per_gpu=16
shuffle1=2
end_type=1
mid_type=1
sca=false
if [ "$sca" = true ]; then
	sca_cmd="--sca"
else
	sca_cmd=""
fi
ch_list=(32)
ch_mid_list=(32)
mid_list=(12 9 6 3)
length=${#ch_list[@]}
n_mids=${#mid_list[@]}

for ((i=0; i<length; i++)); do
	for ((j=0; j<n_mids; j++)); do
		torchrun --nproc_per_node=${n_gpu} --nnodes=1 --rdzv-backend=c10d \
		--rdzv-endpoint=localhost:${dist_port} \
		basicsr/train_param.py --launcher pytorch \
		--template ${template} --num_worker_per_gpu ${num_worker_per_gpu} \
		--batch_size_per_gpu ${batch_size_per_gpu} --num_gpu ${n_gpu} \
		--shuffle1 ${shuffle1}\
		--mid ${mid_list[$j]} --width ${ch_list[$i]} --ch_mid ${ch_mid_list[$i]}
	done
done
