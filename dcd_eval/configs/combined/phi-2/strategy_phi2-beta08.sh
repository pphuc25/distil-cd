#!/bin/bash
python3 src/run_generation.py \
    --model_name_or_path microsoft/phi-2 \
    --max_new_tokens 256 \
    --prompt_file strategyqa \
    --outfile outputs/strategyqa_phi2-beta08.json \
    --cot_flag \
    --constractive_prompt_student 1 \
    --beta_coef 0.8