#!/bin/bash
python3 src/run_generation.py \
    --model_name_or_path meta-llama/Llama-2-7b-hf \
    --max_new_tokens 256 \
    --prompt_file strategyqa \
    --outfile outputs/8bit_strategyqa_llama-7b_cv1-beta08.json \
    --cot_flag \
    --beta_coef 0.8 \
    --quantize_8bit_student \
    --constractive_prompt_student 1