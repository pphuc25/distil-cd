#!/bin/bash
python3 src/run_generation.py \
    --model_name_or_path google/gemma-7b \
    --max_new_tokens 256 \
    --prompt_file strategyqa \
    --outfile outputs/strategyqa_gemma-7b-beta08-drop02.json \
    --cot_flag \
    --constractive_prompt_student 1 \
    --beta_coef 0.8 \
    --dropout_num 0.2