#!/bin/bash
python3 src/run_generation.py \
    --model_name_or_path meta-llama/Llama-2-7b-hf \
    --max_new_tokens 256 \
    --prompt_file gsm8k_500 \
    --student_name_or_path TinyLlama/TinyLlama-1.1B-intermediate-step-1195k-token-2.5T \
    --outfile outputs/gsm8k_new_llama27b_stu1,1b.json \
    --enable_flash_attn2 \
    --cot_flag