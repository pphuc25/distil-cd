#!/bin/bash
python3 src/run_generation.py \
    --model_name_or_path Qwen/Qwen-14B \
    --max_new_tokens 256 \
    --prompt_file gsm8k \
    --outfile outputs/gsm8k_qwen-14b_csprompt_v4-beta03_drop02.json \
    --cot_flag \
    --constractive_prompt_student 4 \
    --dropout_num 0.2 \
    --fp16 \
    --beta_coef 0.3