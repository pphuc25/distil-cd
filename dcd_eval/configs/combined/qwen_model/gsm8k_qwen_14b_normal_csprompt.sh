#!/bin/bash
python3 src/run_generation.py \
    --model_name_or_path Qwen/Qwen-14B \
    --max_new_tokens 256 \
    --prompt_file gsm8k \
    --outfile outputs/gsm8k_full_qwen-14b_csprompt.json \
    --cot_flag \
    --fp16 \
    --use_cs_prompt