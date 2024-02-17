#!/bin/bash
python3 src/run_generation.py \
    --model_name_or_path upstage/SOLAR-10.7B-v1.0 \
    --max_new_tokens 256 \
    --prompt_file strategyqa \
    --outfile outputs/gsm8k_solar-11b_dola_14.json \
    --cot_flag \
    --use_dola \
    --fp16 \
    --early_exit_layers "0,2,4,6,8,10,12,14,48"