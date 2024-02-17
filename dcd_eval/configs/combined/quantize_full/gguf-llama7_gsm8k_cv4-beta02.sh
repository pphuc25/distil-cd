#!/bin/bash
python3 src/run_generation.py \
    --model_name_or_path meta-llama/Llama-2-7b-hf \
    --student_name_or_path TheBloke/Llama-2-7B-GGUF \
    --max_new_tokens 256 \
    --prompt_file gsm8k \
    --outfile outputs/gguf_gsm8k_llama2-7b_cv4-beta02.json \
    --cot_flag \
    --beta_coef 0.2 \
    --constractive_prompt_student 4