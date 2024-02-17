#!/bin/bash
python3 src/run_generation.py \
    --model_name_or_path mistralai/Mistral-7B-v0.1 \
    --max_new_tokens 256 \
    --prompt_file strategyqa \
    --outfile outputs/strategy_full_mistral-7b_csprompt_v1-beta08_drop04.json \
    --cot_flag \
    --constractive_prompt_student 1 \
    --dropout_num 0.4 \
    --beta_coef 0.8