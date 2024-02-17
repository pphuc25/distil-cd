#!/bin/bash
python3 src/run_generation.py \
    --model_name_or_path deepseek-ai/deepseek-llm-7b-base \
    --max_new_tokens 256 \
    --prompt_file strategyqa \
    --outfile outputs/strategyqa_deepseek-7b-base_csprompt.json \
    --cot_flag \
    --use_cs_prompt