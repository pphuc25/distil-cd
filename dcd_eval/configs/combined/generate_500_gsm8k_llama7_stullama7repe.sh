#!/bin/bash
python3 src/run_generation.py \
    --model_name_or_path meta-llama/Llama-2-7b-hf \
    --max_new_tokens 128 \
    --prompt_file gsm8k_500 \
    --outfile outputs/gsm8k_500_llama2-7_repe_v2.json \
    --cot_flag \
    --repe_data_path data/pca_gsm8k.trained_7b_v2 \
    --enable_flash_attn2 \
    --repe_coeff 2.1
