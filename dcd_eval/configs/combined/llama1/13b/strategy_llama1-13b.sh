#!/bin/bash
python3 src/run_generation.py \
    --model_name_or_path huggyllama/llama-13b \
    --max_new_tokens 256 \
    --prompt_file strategyqa \
    --outfile outputs/strategyqa_llama1-13b.json \
    --cot_flag