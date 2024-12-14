# CUDA_VISIBLE_DEVICES=1 uv run python src/classifier/tune.py
experiment_name="llm_financial-roberta-large-sentiment"
CUDA_VISIBLE_DEVICES=1 nohup python src/classifier/tune.py >"nohup_$experiment_name.txt" 2>&1 &
