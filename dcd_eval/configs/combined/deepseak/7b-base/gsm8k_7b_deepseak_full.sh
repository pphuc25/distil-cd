#!/bin/bash
python3 src/run_generation.py \
    --model_name_or_path deepseek-ai/deepseek-llm-7b-base \
    --max_new_tokens 256 \
    --prompt_file gsm8k \
    --outfile outputs/gsm8k-deepseek-7b-base.json \
    --cot_flag