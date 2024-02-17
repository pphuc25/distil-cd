#!/bin/bash
python3 src/run_generation.py \
    --model_name_or_path meta-llama/Llama-2-7b-hf \
    --max_new_tokens 256 \
    --prompt_file gsm8k \
    --outfile outputs/16bit_gsm8k_llama-7b_cv4-beta04-drop03.json \
    --cot_flag \
    --dropout_num 0.3 \
    --beta_coef 0.4 \
    --constractive_prompt_student 4