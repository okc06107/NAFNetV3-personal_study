#!/bin/bash
export CUDA_VISIBLE_DEVICES=7
export PYTHONPATH="$(cd "$(dirname "$0")/../.." && pwd)":$PYTHONPATH

DATA_ROOT="/data1"  # 필요시 /data2 등으로 변경

# === 수동 입력 파라미터 ===
model_path="./experiments/NAFNetV3-RGB-enc0_chmid64_endtype1_mid9_midtype1_shuffle12_width64/models/net_g_latest.pth"
sca=true  # 반드시 직접 입력

# 폴더명에서 파라미터 자동 추출
exp_name=$(echo "$model_path" | grep -oE 'NAFNetV3-RGB-[^/]+')

if [[ $exp_name =~ enc([0-9]+)_chmid([0-9]+)_endtype([0-9]+)_mid([0-9]+)_midtype([0-9]+)_shuffle1([0-9]+)_width([0-9]+) ]]; then
    enc="${BASH_REMATCH[1]}"
    ch_mid="${BASH_REMATCH[2]}"
    end_type="${BASH_REMATCH[3]}"
    mid="${BASH_REMATCH[4]}"
    mid_type="${BASH_REMATCH[5]}"
    shuffle1="${BASH_REMATCH[6]}"
    width="${BASH_REMATCH[7]}"
else
    echo "[ERROR] 파라미터 추출 실패: $exp_name"
    exit 1
fi

# enc 값에 따라 enc_blk_nums, dec_blk_nums 자동 할당
template="options/test/NAFNetV3_RGB/template_rgb_enc${enc}.j2"
case $enc in
    0) enc_blk_nums="[]" ; dec_blk_nums="[]" ;;
    1) enc_blk_nums="[2]" ; dec_blk_nums="[2]" ;;
    2) enc_blk_nums="[2, 2]" ; dec_blk_nums="[2, 2]" ;;
    3) enc_blk_nums="[2, 2, 2]" ; dec_blk_nums="[2, 2, 2]" ;;
    4) enc_blk_nums="[2, 2, 2, 2]" ; dec_blk_nums="[2, 2, 2, 2]" ;;
    *) echo "enc 값 이상"; exit 1 ;;
esac

jinja2 "$template" \
    -D data_root="$DATA_ROOT" \
    -D shuffle1="$shuffle1" -D width="$width" -D ch_mid="$ch_mid" \
    -D enc_blk_nums="$enc_blk_nums" -D mid="$mid" -D dec_blk_nums="$dec_blk_nums" \
    -D end_type="$end_type" -D mid_type="$mid_type" -D sca="$sca" \
    -D pretrain_network_g="$model_path" \
    > temp_config.yml

python isp/run_sidd_benchmark_param.py --template temp_config.yml
rm temp_config.yml