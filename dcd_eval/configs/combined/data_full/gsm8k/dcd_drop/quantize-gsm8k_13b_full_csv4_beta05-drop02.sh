#!/bin/bash
python3 src/run_generation.py \
    --model_name_or_path meta-llama/Llama-2-13b-hf \
    --max_new_tokens 256 \
    --prompt_file gsm8k \
    --outfile outputs/quantize-gsm8k_full_llama2-13b_csprompt_v4-beta05-drop02.json \
    --cot_flag \
    --constractive_prompt_student 4 \
    --beta_coef 0.5 \
    --dropout_num 0.2 \
    --fp16 \
    --quantize_4bit_student