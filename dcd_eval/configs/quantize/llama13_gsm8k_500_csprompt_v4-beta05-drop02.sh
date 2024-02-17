#!/bin/bash
python3 src/run_generation.py \
    --model_name_or_path meta-llama/Llama-2-13b-hf \
    --max_new_tokens 256 \
    --prompt_file gsm8k_500 \
    --outfile outputs/quantize_gsm8k_500_llama2-13b_csprompt_v4_drop02.json \
    --cot_flag \
    --constractive_prompt_student 4 \
    --enable_flash_attn2 \
    --dropout_num 0.2 \
    --quantize_4bit_student