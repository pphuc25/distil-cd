#!/bin/bash
python3 src/run_generation.py \
    --model_name_or_path huggyllama/llama-7b \
    --max_new_tokens 256 \
    --prompt_file strategyqa \
    --outfile outputs/strategyqa_llama1-7b-beta08-drop02.json \
    --cot_flag \
    --constractive_prompt_student 1 \
    --dropout_num 0.2 \
    --beta_coef 0.8