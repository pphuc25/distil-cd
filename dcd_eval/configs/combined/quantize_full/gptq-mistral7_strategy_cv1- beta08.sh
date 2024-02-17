#!/bin/bash
python3 src/run_generation.py \
    --model_name_or_path mistralai/Mistral-7B-v0.1 \
    --student_name_or_path TheBloke/Mistral-7B-v0.1-GPTQ \
    --max_new_tokens 256 \
    --prompt_file strategyqa \
    --outfile outputs/gptq_strategy-7b_cv1-beta08.json \
    --cot_flag \
    --beta_coef 0.8 \
    --constractive_prompt_student 1