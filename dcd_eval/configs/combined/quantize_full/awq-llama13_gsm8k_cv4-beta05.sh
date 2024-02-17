#!/bin/bash
python3 src/run_generation.py \
    --model_name_or_path meta-llama/Llama-2-13b-hf \
    --student_name_or_path TheBloke/LLaMA-13b-AWQ \
    --max_new_tokens 256 \
    --prompt_file gsm8k \
    --outfile outputs/awq_gsm8k_llama2-13b_cv4-beta05.json \
    --cot_flag \
    --beta_coef 0.5 \
    --constractive_prompt_student 4