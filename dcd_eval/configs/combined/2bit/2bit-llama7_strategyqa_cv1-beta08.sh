#!/bin/bash
python3 src/run_generation.py \
    --model_name_or_path meta-llama/Llama-2-7b-hf \
    --student_name_or_path GreenBitAI/LLaMA-7B-2bit \
    --max_new_tokens 256 \
    --prompt_file strategyqa \
    --outfile outputs/2bit_strategyqa_llama-7b_cv1-beta08.json \
    --cot_flag \
    --beta_coef 0.8 \
    --constractive_prompt_student 1