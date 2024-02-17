#!/bin/bash
python3 src/run_generation.py \
    --model_name_or_path upstage/SOLAR-10.7B-v1.0 \
    --max_new_tokens 256 \
    --prompt_file gsm8k \
    --outfile outputs/gsm8k_solar-11b.json \
    --fp16 \
    --cot_flag