# export CUDA_VISIBLE_DEVICES=1

python -u run.py \
  --task_name anomaly_detection \
  --is_training 1 \
  --root_path ./dataset/DACON \
  --model_id DACON \
  --model TimeMixer \
  --data PSMSingle \
  --features M \
  --seq_len 100 \
  --pred_len 0 \
  --downsample 1 \
  --down_sampling_layers 3 \
  --down_sampling_method avg \
  --down_sampling_window 2 \
  --d_model 64 \
  --d_ff 64 \
  --e_layers 3 \
  --enc_in 1 \
  --c_out 1 \
  --top_k 3 \
  --anomaly_ratio 1 \
  --batch_size 64 \
  --train_epochs 10 \
  --patience 10