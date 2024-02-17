#!/bin/bash
python3 src/run_generation.py \
    --model_name_or_path meta-llama/Llama-2-7b-hf \
    --max_new_tokens 256 \
    --prompt_file strategyqa \
    --outfile outputs/quantize-strategy_full_llama2-7b_csprompt_v1-beta05_drop02.json \
    --cot_flag \
    --constractive_prompt_student 1 \
    --beta_coef 0.5 \
    --dropout_num 0.2 \
    --quantize_4bit_student