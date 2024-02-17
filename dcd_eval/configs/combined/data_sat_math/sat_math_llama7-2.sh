#!/bin/bash
python3 src/run_generation.py \
    --model_name_or_path meta-llama/Llama-2-7b-hf \
    --max_new_tokens 256 \
    --prompt_file sat_math \
    --outfile outputs/sat_math_llama2-7b.json \
    --cot_flag