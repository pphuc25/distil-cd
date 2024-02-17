#!/bin/bash
python3 src/run_generation.py \
    --model_name_or_path huggyllama/llama-13b \
    --max_new_tokens 256 \
    --prompt_file gsm8k \
    --outfile outputs/gsm8k_llama1-13b-beta05-drop02.json \
    --cot_flag \
    --constractive_prompt_student 4 \
    --dropout_num 0.2 \
    --beta_coef 0.5