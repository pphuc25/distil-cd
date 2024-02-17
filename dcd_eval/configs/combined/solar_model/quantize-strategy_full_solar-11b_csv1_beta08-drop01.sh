#!/bin/bash
python3 src/run_generation.py \
    --model_name_or_path upstage/SOLAR-10.7B-v1.0 \
    --max_new_tokens 256 \
    --prompt_file strategyqa \
    --outfile outputs/quantize-strategy_full_solar-11b_csv1-beta08-drop01.json \
    --cot_flag \
    --constractive_prompt_student 1 \
    --beta_coef 0.8 \
    --dropout_num 0.1 \
    --fp16 \
    --quantize_4bit_student