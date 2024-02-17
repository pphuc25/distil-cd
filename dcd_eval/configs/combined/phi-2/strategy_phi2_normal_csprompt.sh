#!/bin/bash
python3 src/run_generation.py \
    --model_name_or_path microsoft/phi-2 \
    --max_new_tokens 256 \
    --prompt_file strategyqa \
    --outfile outputs/strategyqa_phi2_csprompt.json \
    --cot_flag \
    --use_cs_prompt