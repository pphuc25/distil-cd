#!/bin/bash
python3 src/run_generation.py \
    --model_name_or_path meta-llama/Llama-2-7b-hf \
    --max_new_tokens 256 \
    --prompt_file sat_math \
    --outfile outputs/sat_math_llama2-7b-dola.json \
    --cot_flag \
    --constractive_prompt_student 1 \
    --use_dola \
    --early_exit_layers "0,2,4,6,8,10,12,14,32"