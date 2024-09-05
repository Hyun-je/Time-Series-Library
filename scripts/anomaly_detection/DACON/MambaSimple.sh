# export CUDA_VISIBLE_DEVICES=1

python -u run.py \
  --task_name anomaly_detection \
  --is_training 1 \
  --root_path ./dataset/DACON \
  --model_id DACON \
  --model MambaSimple \
  --data PSM \
  --features M \
  --seq_len 125 \
  --pred_len 0 \
  --downsample 8 \
  --d_model 128 \
  --d_ff 128 \
  --e_layers 3 \
  --enc_in 51 \
  --c_out 51 \
  --output_attention \
  --dropout 0.2 \
  --top_k 3 \
  --anomaly_ratio 0.8 \
  --seed 12345678 \
  --batch_size 64 \
  --train_epochs 1 \
  --patience 15