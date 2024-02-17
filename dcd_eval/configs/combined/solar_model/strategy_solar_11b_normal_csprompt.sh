#!/bin/bash
python3 src/run_generation.py \
    --model_name_or_path upstage/SOLAR-10.7B-v1.0 \
    --max_new_tokens 256 \
    --prompt_file strategyqa \
    --outfile outputs/strategyqa_full_solor-11b_csprompt.json \
    --cot_flag \
    --fp16 \
    --use_cs_prompt