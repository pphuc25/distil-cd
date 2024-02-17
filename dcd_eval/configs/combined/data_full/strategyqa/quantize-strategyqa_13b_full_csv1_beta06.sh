#!/bin/bash
python3 src/run_generation.py \
    --model_name_or_path meta-llama/Llama-2-13b-hf \
    --max_new_tokens 256 \
    --prompt_file strategyqa \
    --outfile outputs/quantize-strategy_full_llama2-13b_csprompt_v1-beta06.json \
    --cot_flag \
    --constractive_prompt_student 1 \
    --beta_coef 0.6 \
    --dropout_num 0.2 \
    --quantize_4bit_student