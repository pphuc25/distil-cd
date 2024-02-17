#!/bin/bash
python3 src/run_generation.py \
    --model_name_or_path meta-llama/Llama-2-7b-hf \
    --max_new_tokens 256 \
    --prompt_file gsm8k \
    --outfile outputs/quantize-gsm8k_full_llama2-7b_csprompt_v4-beta02-drop02.json \
    --cot_flag \
    --constractive_prompt_student 4 \
    --beta_coef 0.2 \
    --beta_coef 0.2 \
    --quantize_4bit_student