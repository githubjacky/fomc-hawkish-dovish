# CUDA_VISIBLE_DEVICES=1 uv run python src/classifier/tune.py
CUDA_VISIBLE_DEVICES=1 nohup uv run python src/classifier/tune.py >"nohup.txt" 2>&1 &
