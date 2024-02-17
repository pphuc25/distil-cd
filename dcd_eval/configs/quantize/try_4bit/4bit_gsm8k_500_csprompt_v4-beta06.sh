#!/bin/bash
python3 src/run_generation.py \
    --model_name_or_path meta-llama/Llama-2-7b-hf \
    --max_new_tokens 256 \
    --prompt_file gsm8k_500 \
    --outfile outputs/quantize_gsm8k_500_llama2-7b_csprompt_v4-beta06.json \
    --cot_flag \
    --constractive_prompt_student 4 \
    --enable_flash_attn2 \
    --beta_coef 0.6 \
    --quantize_4bit_student
