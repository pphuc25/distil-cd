#!/bin/bash
python3 src/run_generation.py \
    --model_name_or_path meta-llama/Llama-2-7b-hf \
    --max_new_tokens 256 \
    --prompt_file strategyqa \
    --outfile outputs/strategy_full_llama2-7b_csprompt_v1-beta09_drop01.json \
    --cot_flag \
    --constractive_prompt_student 1 \
    --dropout_num 0.1 \
    --beta_coef 0.9