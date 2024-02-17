#!/bin/bash
python3 src/run_generation.py \
    --model_name_or_path upstage/SOLAR-10.7B-v1.0 \
    --max_new_tokens 256 \
    --prompt_file gsm8k \
    --outfile outputs/quantize-gsm8k_full_solar-11b_csprompt_v4-beta07_drop01.json \
    --cot_flag \
    --constractive_prompt_student 4 \
    --beta_coef 0.7 \
    --dropout_num 0.1 \
    --fp16 \
    --quantize_4bit_student