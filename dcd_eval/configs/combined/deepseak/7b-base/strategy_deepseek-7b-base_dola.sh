#!/bin/bash
python3 src/run_generation.py \
    --model_name_or_path deepseek-ai/deepseek-llm-7b-base \
    --max_new_tokens 256 \
    --prompt_file strategyqa \
    --outfile outputs/strategyqa_deepseek-7b-base_dola.json \
    --cot_flag \
    --use_dola \
    --early_exit_layers "0,2,4,6,8,10,12,14,30"