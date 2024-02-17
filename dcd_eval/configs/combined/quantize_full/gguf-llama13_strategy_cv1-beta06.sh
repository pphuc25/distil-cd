#!/bin/bash
python3 src/run_generation.py \
    --model_name_or_path meta-llama/Llama-2-13b-hf \
    --student_name_or_path TheBloke/Llama-2-13B-GGUF \
    --max_new_tokens 256 \
    --prompt_file strategyqa \
    --outfile outputs/gguf_strategy_llama2-13b_cv1-beta06.json \
    --beta_coef 0.6 \
    --cot_flag \
    --constractive_prompt_student 4