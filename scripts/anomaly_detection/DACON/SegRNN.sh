# export CUDA_VISIBLE_DEVICES=6

python -u run.py \
  --task_name anomaly_detection \
  --is_training 1 \
  --root_path ./dataset/DACON \
  --model_id DACON \
  --model SegRNN \
  --data PSM \
  --features M \
  --seq_len 720 \
  --pred_len 0 \
  --seg_len 48 \
  --downsample 1 \
  --d_model 512 \
  --d_ff 256 \
  --e_layers 3 \
  --enc_in 51 \
  --c_out 51 \
  --dropout 0.1 \
  --top_k 3 \
  --anomaly_ratio 1 \
  --batch_size 256 \
  --train_epochs 10 \
  --loss MSE \
  --patience 15