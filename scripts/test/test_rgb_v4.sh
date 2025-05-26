#!/bin/bash
export CUDA_VISIBLE_DEVICES=7
export PYTHONPATH="$(cd "$(dirname "$0")/../.." && pwd)":$PYTHONPATH

RESULTS_DIR="auto_val_result"
mkdir -p "$RESULTS_DIR"

DATA_ROOT="/data1"  # 필요시 /data2 등으로 변경
sca=true  # 반드시 직접 입력 (true/false)

result_file="$RESULTS_DIR/total_result.txt"
> "$result_file"  # 파일 초기화

for exp_dir in ./experiments/NAFNetV3-RGB-*; do
    [ -d "$exp_dir" ] || continue
    exp_name=$(echo "$exp_dir" | grep -oE 'NAFNetV3-RGB-[^/]+')

    # 폴더명에서 파라미터 추출
    if [[ $exp_name =~ enc([0-9]+)_chmid([0-9]+)_endtype([0-9]+)_mid([0-9]+)_midtype([0-9]+)_shuffle1([0-9]+)_width([0-9]+) ]]; then
        enc="${BASH_REMATCH[1]}"
        ch_mid="${BASH_REMATCH[2]}"
        end_type="${BASH_REMATCH[3]}"
        mid="${BASH_REMATCH[4]}"
        mid_type="${BASH_REMATCH[5]}"
        shuffle1="${BASH_REMATCH[6]}"
        width="${BASH_REMATCH[7]}"
    else
        echo "[ERROR] 파라미터 추출 실패: $exp_name" | tee -a "$RESULTS_DIR/error.txt"
        continue
    fi

    # enc별 템플릿 선택
    template="options/test/NAFNetV3_RGB/template_rgb_enc${enc}.j2"

    # 모델 체크
    model_path="$exp_dir/models/net_g_latest.pth"
    if [ ! -f "$model_path" ]; then
        echo "[ERROR] 모델 없음: $model_path" | tee -a "$RESULTS_DIR/error.txt"
        continue
    fi

    # enc별 파라미터
    case $enc in
        0) enc_blk_nums="[]" ; dec_blk_nums="[]" ; mid_type="1" ; end_type="1" ;;
        1) enc_blk_nums="[2]" ; dec_blk_nums="[2]" ; mid_type="1" ; end_type="1" ;;
        2) enc_blk_nums="[2, 2]" ; dec_blk_nums="[2, 2]" ; mid_type="0" ; end_type="0" ;;
        3) enc_blk_nums="[2, 2, 2]" ; dec_blk_nums="[2, 2, 2]" ; mid_type="0" ; end_type="0" ;;
        4) enc_blk_nums="[2, 2, 2, 2]" ; dec_blk_nums="[2, 2, 2, 2]" ; mid_type="0" ; end_type="0" ;;
        *) echo "[ERROR] enc값 이상: $enc" | tee -a "$RESULTS_DIR/error.txt"; continue ;;
    esac

    pretrain_network_g="$model_path"

    # 템플릿 렌더링 (jinja2-cli 필요)
    jinja2 "$template" \
        -D data_root="$DATA_ROOT" \
        -D enc="$enc" -D shuffle1="$shuffle1" -D width="$width" -D ch_mid="$ch_mid" \
        -D enc_blk_nums="$enc_blk_nums" -D mid="$mid" -D dec_blk_nums="$dec_blk_nums" \
        -D end_type="$end_type" -D mid_type="$mid_type" -D sca="$sca" \
        -D pretrain_network_g="$pretrain_network_g" \
        > "temp_config.yml"

    echo "[${exp_name}]" >> "$result_file"
    python isp/run_sidd_benchmark_param.py --template temp_config.yml 2>&1 >> "$result_file"
    echo -e "\n-----------------------------\n" >> "$result_file"

    rm temp_config.yml
done 