# CUDA_VISIBLE_DEVICES=1 uv run python src/classifier/tune.py
experiment_name="finetune_pooler_output_twitter-roberta-base-sentiment-latest"
CUDA_VISIBLE_DEVICES=1 nohup python src/classifier/tune.py >"nohup_$experiment_name.txt" 2>&1 &
