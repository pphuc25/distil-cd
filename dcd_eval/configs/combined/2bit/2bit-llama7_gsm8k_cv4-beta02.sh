#!/bin/bash
python3 src/run_generation.py \
    --model_name_or_path meta-llama/Llama-2-7b-hf \
    --student_name_or_path GreenBitAI/LLaMA-7B-2bit \
    --max_new_tokens 256 \
    --prompt_file gsm8k \
    --outfile outputs/2bit_gsm8k_llama-7b_cv4-beta02.json \
    --cot_flag \
    --beta_coef 0.2 \
    --constractive_prompt_student 4