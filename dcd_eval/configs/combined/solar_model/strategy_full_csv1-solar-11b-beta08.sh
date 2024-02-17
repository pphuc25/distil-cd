#!/bin/bash
python3 src/run_generation.py \
    --model_name_or_path upstage/SOLAR-10.7B-v1.0 \
    --max_new_tokens 256 \
    --prompt_file strategyqa \
    --outfile outputs/strategy_solar-11b_csv1-beta08.json \
    --cot_flag \
    --constractive_prompt_student 1 \
    --fp16 \
    --beta_coef 0.8