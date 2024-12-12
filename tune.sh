# CUDA_VISIBLE_DEVICES=1 uv run python src/classifier/tune.py
experiment_name="cls_pooling_sbert"
CUDA_VISIBLE_DEVICES=0 nohup python src/classifier/tune.py >"nohup_$experiment_name.txt" 2>&1 &
