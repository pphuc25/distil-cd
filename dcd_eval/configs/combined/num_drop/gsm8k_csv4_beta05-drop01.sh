#!/bin/bash
python3 src/run_generation.py \
    --model_name_or_path meta-llama/Llama-2-7b-hf \
    --max_new_tokens 256 \
    --prompt_file gsm8k_500 \
    --outfile outputs/gsm8k_llama2-7b_csprompt_v4-beta05_drop01.json \
    --cot_flag \
    --constractive_prompt_student 4 \
    --dropout_num 0.1 \
    --beta_coef 0.5 \
    --enable_flash_attn2