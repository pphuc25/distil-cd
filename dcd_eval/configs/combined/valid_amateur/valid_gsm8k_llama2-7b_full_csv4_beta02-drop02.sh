#!/bin/bash
python3 src/run_generation.py \
    --model_name_or_path meta-llama/Llama-2-7b-hf \
    --max_new_tokens 256 \
    --prompt_file gsm8k \
    --outfile outputs/valid_amateur_gsm8k_full_llama2-7b_csprompt_v4-beta02-drop02.json \
    --cot_flag \
    --valid_amateur \
    --beta_coef 0.2 \
    --dropout_num 0.2