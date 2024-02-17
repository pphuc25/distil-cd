#!/bin/bash
python3 src/run_generation.py \
    --model_name_or_path mistralai/Mistral-7B-v0.1 \
    --max_new_tokens 256 \
    --prompt_file gsm8k_500 \
    --outfile outputs/gsm8k_mistral-7b_csprompt_v4-beta06_drop02.json \
    --cot_flag \
    --constractive_prompt_student 4 \
    --dropout_num 0.2 \
    --beta_coef 0.6 \
    --enable_flash_attn2
