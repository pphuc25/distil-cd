#!/bin/bash
python3 src/run_generation.py \
    --model_name_or_path meta-llama/Llama-2-13b-hf \
    --student_name_or_path TheBloke/Llama-2-13B-GPTQ \
    --max_new_tokens 256 \
    --prompt_file strategyqa \
    --outfile outputs/gptq_strategy_llama2-13b_cv1-beta06.json \
    --cot_flag \
    --beta_coef 0.6 \
    --constractive_prompt_student 1