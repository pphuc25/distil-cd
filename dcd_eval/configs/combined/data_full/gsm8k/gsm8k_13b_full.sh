#!/bin/bash
python3 src/run_generation.py \
    --model_name_or_path meta-llama/Llama-2-13b-hf \
    --max_new_tokens 256 \
    --prompt_file gsm8k \
    --outfile outputs/gsm8k-13b.json \
    --cot_flag