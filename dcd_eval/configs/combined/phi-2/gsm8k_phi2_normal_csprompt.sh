#!/bin/bash
python3 src/run_generation.py \
    --model_name_or_path microsoft/phi-2 \
    --max_new_tokens 256 \
    --prompt_file gsm8k \
    --outfile outputs/gsm8k_phi2_csprompt.json \
    --cot_flag \
    --use_cs_prompt