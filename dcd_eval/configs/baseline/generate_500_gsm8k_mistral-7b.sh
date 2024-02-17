#!/bin/bash
python3 src/run_generation.py \
    --model_name_or_path mistralai/Mistral-7B-v0.1 \
    --max_new_tokens 256 \
    --prompt_file gsm8k_500 \
    --outfile outputs/gsm8k_500_new_mistral-7b.json \
    --enable_flash_attn2 \
    --cot_flag