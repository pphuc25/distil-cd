#!/bin/bash
python3 src/run_generation.py \
    --model_name_or_path meta-llama/Llama-2-7b-hf \
    --max_new_tokens 256 \
    --prompt_file strategyqa_500 \
    --outfile outputs/strategy_llama2-7b_csprompt_v4-beta09_drop02.json \
    --cot_flag \
    --constractive_prompt_student 1 \
    --dropout_num 0.2 \
    --beta_coef 0.9 \
    --enable_flash_attn2