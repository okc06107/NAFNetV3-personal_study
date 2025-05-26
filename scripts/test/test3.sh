export CUDA_VISIBLE_DEVICES=6
template="options/test/NAFNetV3/template_baseline_enc3.j2"

shuffle1=2
ch_list=(32)
ch_mid_list=(32)
mid_list=(12)
length=${#ch_list[@]}
n_mids=${#mid_list[@]}

for ((i=0; i<length; i++)); do
	for ((j=0; j<n_mids; j++)); do
		python isp/run_sidd_benchmark_param.py --template ${template} \
		--shuffle1 ${shuffle1}\
		--mid ${mid_list[$j]} --width ${ch_list[$i]} --ch_mid ${ch_mid_list[$i]}
	done
done

