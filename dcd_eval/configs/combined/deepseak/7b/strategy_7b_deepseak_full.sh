#!/bin/bash
python3 src/run_generation.py \
    --model_name_or_path deepseek-ai/deepseek-math-7b-base \
    --max_new_tokens 256 \
    --prompt_file strategyqa \
    --outfile outputs/strategyqa-deepseek-7b.json \
    --cot_flag