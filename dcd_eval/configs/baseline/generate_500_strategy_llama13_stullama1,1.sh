#!/bin/bash
python3 src/run_generation.py \
    --model_name_or_path meta-llama/Llama-2-13b-hf \
    --max_new_tokens 256 \
    --prompt_file strategyqa_500 \
    --student_name_or_path TinyLlama/TinyLlama-1.1B-intermediate-step-1195k-token-2.5T \
    --outfile outputs/strategyqa_new_llama213b_stu1,1b.json \
    --cot_flag