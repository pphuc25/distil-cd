#!/bin/bash
python3 src/run_generation.py \
    --model_name_or_path Qwen/Qwen-14B \
    --max_new_tokens 256 \
    --prompt_file gsm8k \
    --outfile outputs/gsm8k_qwen-14b_dola.json \
    --cot_flag \
    --use_dola \
    --fp16 \
    --early_exit_layers "0,2,4,6,8,10,12,14,16,18,40"