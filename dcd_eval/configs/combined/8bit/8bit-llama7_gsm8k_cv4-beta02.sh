#!/bin/bash
python3 src/run_generation.py \
    --model_name_or_path meta-llama/Llama-2-7b-hf \
    --max_new_tokens 256 \
    --prompt_file gsm8k \
    --outfile outputs/8bit_gsm8k_llama-7b_cv4-beta02.json \
    --cot_flag \
    --beta_coef 0.2 \
    --quantize_8bit_student \
    --constractive_prompt_student 4