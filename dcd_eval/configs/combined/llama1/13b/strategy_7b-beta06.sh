#!/bin/bash
python3 src/run_generation.py \
    --model_name_or_path huggyllama/llama-13b \
    --max_new_tokens 256 \
    --prompt_file strategyqa \
    --outfile outputs/strategyqa_llama1-13b-beta06.json \
    --cot_flag \
    --constractive_prompt_student 4 \
    --beta_coef 0.6