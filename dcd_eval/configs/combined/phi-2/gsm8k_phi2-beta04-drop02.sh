#!/bin/bash
python3 src/run_generation.py \
    --model_name_or_path microsoft/phi-2 \
    --max_new_tokens 256 \
    --prompt_file gsm8k \
    --outfile outputs/gsm8k_phi2-beta04-drop02.json \
    --cot_flag \
    --constractive_prompt_student 4 \
    --beta_coef 0.4 \
    --dropout_num 0.2