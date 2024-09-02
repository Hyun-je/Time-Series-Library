# export CUDA_VISIBLE_DEVICES=6

python -u run.py \
  --task_name anomaly_detection \
  --is_training 1 \
  --root_path ./dataset/DACON \
  --model_id DACON \
  --model DLinear \
  --data PSM \
  --features M \
  --seq_len 100 \
  --moving_avg 25 \
  --downsample 1 \
  --enc_in 51 \
  --c_out 51 \
  --anomaly_ratio 1 \
  --batch_size 256 \
  --train_epochs 10 \
  --loss MSE \
  --patience 15