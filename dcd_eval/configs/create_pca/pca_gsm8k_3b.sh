python3 make_PCA_data.py \
    --model_name openlm-research/open_llama_3b_v2 \
    --prompt_right "Pretend to be a math teacher having IMO gold medals" \
    --prompt_wrong "Pretend to be a elementary student with lower academic performance" \
    --output_path "data/pca_gsm8k.trained_3b"