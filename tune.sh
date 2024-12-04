# CUDA_VISIBLE_DEVICES=1 uv run python src/classifier/tune.py
CUDA_VISIBLE_DEVICES=2 nohup uv run python src/classifier/tune.py >"nohup_gru_real.txt" 2>&1 &
