#!/bin/bash
python3 src/run_generation.py \
    --model_name_or_path meta-llama/Llama-2-7b-hf \
    --max_new_tokens 256 \
    --prompt_file strategyqa \
    --outfile outputs/valid_amateur_strategyqa_full_llama2-7b_csprompt_v4-beta08.json \
    --cot_flag \
    --valid_amateur \
    --beta_coef 0.8