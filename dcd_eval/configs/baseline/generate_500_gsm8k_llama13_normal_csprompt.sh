#!/bin/bash
python3 src/run_generation.py \
    --model_name_or_path meta-llama/Llama-2-13b-hf \
    --max_new_tokens 256 \
    --prompt_file gsm8k_500 \
    --outfile outputs/gsm8k_500_new_llama2-13b_csprompt.json \
    --cot_flag \
    --enable_flash_attn2 \
    --use_cs_prompt