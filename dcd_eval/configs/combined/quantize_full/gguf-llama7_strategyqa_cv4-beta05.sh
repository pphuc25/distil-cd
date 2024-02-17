#!/bin/bash
python3 src/run_generation.py \
    --model_name_or_path meta-llama/Llama-2-7b-hf \
    --student_name_or_path TheBloke/Llama-2-7B-GGUF \
    --max_new_tokens 256 \
    --prompt_file strategyqa \
    --outfile outputs/gguf_gsm8k_llama2-7b_cv1-beta05.json \
    --beta_coef 0.5 \
    --cot_flag \
    --constractive_prompt_student 1