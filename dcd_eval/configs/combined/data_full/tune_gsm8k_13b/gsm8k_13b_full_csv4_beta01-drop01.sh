#!/bin/bash
python3 src/run_generation.py \
    --model_name_or_path meta-llama/Llama-2-13b-hf \
    --max_new_tokens 256 \
    --prompt_file gsm8k \
    --outfile outputs/gsm8k_full_llama2-13b_csprompt_v4-beta01-drop01.json \
    --cot_flag \
    --constractive_prompt_student 4 \
    --dropout_num 0.1 \
    --fp16 \
    --beta_coef 0.1