# CUDA_VISIBLE_DEVICES=1 uv run python src/classifier/tune.py
experiment_name="h1_multi-qa-mpnet-base-dot-v1"
CUDA_VISIBLE_DEVICES=1 nohup python src/classifier/tune.py >"nohup_$experiment_name.txt" 2>&1 &
