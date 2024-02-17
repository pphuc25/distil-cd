#!/bin/bash
python3 src/run_generation.py \
    --model_name_or_path Qwen/Qwen-14B \
    --max_new_tokens 256 \
    --prompt_file strategyqa \
    --outfile outputs/quantize-strategy_full_qwen-14b_csprompt_v1-beta07_drop01.json \
    --cot_flag \
    --constractive_prompt_student 1 \
    --beta_coef 0.7 \
    --dropout_num 0.1 \
    --fp16 \
    --quantize_4bit_student