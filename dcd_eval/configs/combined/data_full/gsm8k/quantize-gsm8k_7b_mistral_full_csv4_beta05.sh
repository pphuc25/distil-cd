#!/bin/bash
python3 src/run_generation.py \
    --model_name_or_path mistralai/Mistral-7B-v0.1 \
    --max_new_tokens 256 \
    --prompt_file gsm8k \
    --outfile outputs/quantize-gsm8k_full_mistral-7b_csprompt_v4-beta05.json \
    --cot_flag \
    --constractive_prompt_student 4 \
    --beta_coef 0.5 \
    --quantize_4bit_student