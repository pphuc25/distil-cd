#!/bin/bash
python3 src/run_generation.py \
    --model_name_or_path meta-llama/Llama-2-7b-hf \
    --max_new_tokens 256 \
    --prompt_file gsm8k_500 \
    --outfile outputs/gsm8k_new_llama27b_dola_csprompt.json \
    --cot_flag \
    --constractive_prompt_student 2 \
    --use_dola \
    --enable_flash_attn2 \
    --theta_coef 0.5 \
    --early_exit_layers "0,2,4,6,8,10,12,14,32"