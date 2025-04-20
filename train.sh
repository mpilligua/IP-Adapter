#!/bin/bash
#SBATCH -A dep # account
#SBATCH -n 8 # Number of cores
#SBATCH -N 1 # Ensure that all cores are on one machine
#SBATCH -D /ghome/mpilligua/TFG/IP-Adapter
#SBATCH -p dcca40 # Partition to submit to
#SBATCH --mem 8000 # 2GB solicitados.
#SBATCH -o %x_%j.out # File to which STDOUT will be written
#SBATCH -e %x_%j.err # File to which STDERR will be written
#SBATCH --gres gpu:4 # Para pedir gráficas

cd /ghome/mpilligua/TFG/IP-Adapter

accelerate launch --num_processes 4 --multi_gpu --mixed_precision "fp16" --main_process_port 29550\
  tutorial_train.py \
  --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
  --image_encoder_path="/ghome/mpilligua/TFG/IP-Adapter/models/image_encoder/" \
  --data_json_file="/ghome/mpilligua/TFG/IP-Adapter/data_jsons/baseline.json" \
  --data_root_path="/ghome/mpilligua/TFG/Datasets/RSR_256" \
  --mixed_precision="fp16" \
  --resolution=512 \
  --train_batch_size=32 \
  --dataloader_num_workers=4 \
  --learning_rate=1e-04 \
  --weight_decay=0.01 \
  --output_dir="/ghome/mpilligua/TFG/IP-Adapter/trainings/" \
  --save_steps=500 \
  --num_train_epochs=2000 \
  --pretrained_ip_adapter_path="/ghome/mpilligua/TFG/IP-Adapter/trainings/2025-04-19_14-23-17/checkpoint-300/ip_adapter.bin"