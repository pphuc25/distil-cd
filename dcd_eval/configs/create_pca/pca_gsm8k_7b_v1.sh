python3 make_PCA_data.py \
    --model_name meta-llama/Llama-2-7b-hf \
    --prompt_right "Pretend to be a math teacher having IMO gold medals" \
    --prompt_wrong "Pretend to be a elementary student with lower academic performance" \
    --output_path "data/pca_gsm8k.trained_7b_v1"