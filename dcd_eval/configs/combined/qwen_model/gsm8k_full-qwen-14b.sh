#!/bin/bash
python3 src/run_generation.py \
    --model_name_or_path Qwen/Qwen-14B \
    --max_new_tokens 256 \
    --prompt_file gsm8k \
    --outfile outputs/gsm8k_qwen-14b.json \
    --fp16 \
    --cot_flag