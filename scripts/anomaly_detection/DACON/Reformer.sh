# export CUDA_VISIBLE_DEVICES=1

python -u run.py \
  --task_name anomaly_detection \
  --is_training 1 \
  --root_path ./dataset/DACON \
  --model_id DACON \
  --model Reformer \
  --data PSM \
  --features M \
  --seq_len 100 \
  --pred_len 0 \
  --downsample 1 \
  --d_model 128 \
  --d_ff 128 \
  --e_layers 3 \
  --enc_in 51 \
  --c_out 51 \
  --dropout 0.1 \
  --top_k 3 \
  --anomaly_ratio 1 \
  --batch_size 256 \
  --train_epochs 10 \
  --patience 15