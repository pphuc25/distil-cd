#!/bin/bash
python3 src/run_generation.py \
    --model_name_or_path meta-llama/Llama-2-13b-hf \
    --max_new_tokens 256 \
    --prompt_file strategyqa_500 \
    --outfile outputs/quantize_strategyqa_500_llama2-13b_csprompt_v1-beta05.json \
    --cot_flag \
    --constractive_prompt_student 1 \
    --enable_flash_attn2 \
    --quantize_4bit_student