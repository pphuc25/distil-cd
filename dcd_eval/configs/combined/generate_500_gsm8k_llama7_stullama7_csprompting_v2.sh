#!/bin/bash
python3 src/run_generation.py \
    --model_name_or_path meta-llama/Llama-2-7b-hf \
    --max_new_tokens 256 \
    --prompt_file gsm8k_500 \
    --outfile outputs/gsm8k_500_new_llama2-7b_csprompt_v2.json \
    --cot_flag \
    --enable_flash_attn2 \
    --constractive_prompt_student 2