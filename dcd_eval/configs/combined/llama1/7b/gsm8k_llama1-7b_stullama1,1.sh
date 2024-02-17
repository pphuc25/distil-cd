#!/bin/bash
python3 src/run_generation.py \
    --model_name_or_path huggyllama/llama-7b \
    --max_new_tokens 256 \
    --prompt_file gsm8k \
    --student_name_or_path TinyLlama/TinyLlama-1.1B-intermediate-step-1195k-token-2.5T \
    --outfile outputs/gsm8k_llama1-7b_stu1,1b.json \
    --cot_flag