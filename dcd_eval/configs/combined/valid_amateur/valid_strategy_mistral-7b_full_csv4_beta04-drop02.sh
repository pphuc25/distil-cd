#!/bin/bash
python3 src/run_generation.py \
    --model_name_or_path mistralai/Mistral-7B-v0.1 \
    --max_new_tokens 256 \
    --prompt_file strategyqa \
    --outfile outputs/valid_amateur_strategyqa_full_mistral-7b_csprompt_v4-beta08-drop02.json \
    --cot_flag \
    --valid_amateur \
    --beta_coef 0.8 \
    --dropout_num 0.2