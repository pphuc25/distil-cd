#!/bin/bash
python3 src/run_generation.py \
    --model_name_or_path deepseek-ai/deepseek-moe-16b-base \
    --max_new_tokens 256 \
    --prompt_file strategyqa \
    --outfile outputs/strategyqa-deepseek-16b-drop08-beta02.json \
    --cot_flag \
    --constractive_prompt_student 1 \
    --dropout_num 0.2 \
    --beta_coef 0.8