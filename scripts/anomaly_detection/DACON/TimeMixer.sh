# export CUDA_VISIBLE_DEVICES=1

python -u run.py \
  --task_name anomaly_detection \
  --is_training 1 \
  --root_path ./dataset/DACON \
  --model_id DACON \
  --model TimeMixer \
  --data PSM \
  --features M \
  --seq_len 100 \
  --pred_len 0 \
  --downsample 1 \
  --down_sampling_layers 3 \
  --down_sampling_method avg \
  --down_sampling_window 2 \
  --d_model 128 \
  --d_ff 128 \
  --e_layers 3 \
  --enc_in 51 \
  --c_out 51 \
  --top_k 3 \
  --anomaly_ratio 1 \
  --batch_size 256 \
  --train_epochs 30 \
  --patience 10