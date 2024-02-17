#!/bin/bash
python3 src/run_generation.py \
    --model_name_or_path huggyllama/llama-13b \
    --max_new_tokens 256 \
    --prompt_file gsm8k \
    --outfile outputs/gsm8k_llama1-13b-beta05.json \
    --cot_flag \
    --constractive_prompt_student 4 \
    --beta_coef 0.5