#!/bin/bash
python3 src/run_generation.py \
    --model_name_or_path deepseek-ai/deepseek-moe-16b-base \
    --max_new_tokens 256 \
    --prompt_file strategyqa \
    --outfile outputs/gsm8k-deepseek-16b.json \
    --cot_flag