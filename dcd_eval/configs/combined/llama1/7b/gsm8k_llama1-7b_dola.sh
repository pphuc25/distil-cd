#!/bin/bash
python3 src/run_generation.py \
    --model_name_or_path huggyllama/llama-7b \
    --max_new_tokens 256 \
    --prompt_file gsm8k \
    --outfile outputs/gsm8k_llama1-7b_dola.json \
    --cot_flag \
    --use_dola \
    --early_exit_layers "0,2,4,6,8,10,12,14,32"