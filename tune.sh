# CUDA_VISIBLE_DEVICES=1 uv run python src/classifier/tune.py
experiment_name="finance-embeddings-investopedia"
CUDA_VISIBLE_DEVICES=2 nohup python src/classifier/tune.py >"nohup_$experiment_name.txt" 2>&1 &
