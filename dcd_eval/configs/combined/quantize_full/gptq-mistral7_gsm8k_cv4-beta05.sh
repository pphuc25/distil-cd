#!/bin/bash
python3 src/run_generation.py \
    --model_name_or_path mistralai/Mistral-7B-v0.1 \
    --student_name_or_path TheBloke/Mistral-7B-v0.1-GPTQ \
    --max_new_tokens 256 \
    --prompt_file gsm8k \
    --outfile outputs/gptq_gsm8k-7b_cv4-beta05.json \
    --beta_coef 0.5 \
    --cot_flag \
    --constractive_prompt_student 4