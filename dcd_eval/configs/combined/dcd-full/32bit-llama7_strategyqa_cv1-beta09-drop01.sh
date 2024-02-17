#!/bin/bash
python3 src/run_generation.py \
    --model_name_or_path meta-llama/Llama-2-7b-hf \
    --max_new_tokens 256 \
    --prompt_file strategyqa \
    --outfile outputs/16bit_strategyqa_llama-7b_cv1-beta09-drop01.json \
    --cot_flag \
    --dropout_num 0.1 \
    --beta_coef 0.9 \
    --constractive_prompt_student 1