#!/bin/bash
#SBATCH -A dep # account
#SBATCH -n 8 # Number of cores
#SBATCH -N 1 # Ensure that all cores are on one machine
#SBATCH -D /ghome/mpilligua/TFG/IP-Adapter
#SBATCH -p dcca40 # Partition to submit to
#SBATCH --mem 8000 # 2GB solicitados.
#SBATCH -o %x_%j.out # File to which STDOUT will be written
#SBATCH -e %x_%j.err # File to which STDERR will be written
#SBATCH --gres gpu:2 # Para pedir gráficas

cd /ghome/mpilligua/TFG/IP-Adapter

accelerate launch --num_processes 2 --multi_gpu --mixed_precision "fp16" \
  tutorial_train_sdxl.py \
  --pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0" \
  --image_encoder_path="/ghome/mpilligua/TFG/IP-Adapter/sdxl_models/image_encoder/" \
  --data_json_file="/ghome/mpilligua/TFG/IP-Adapter/data_jsons/baseline.json" \
  --data_root_path="/ghome/mpilligua/TFG/Datasets/RSR_256" \
  --mixed_precision="fp16" \
  --resolution=512 \
  --train_batch_size=8 \
  --dataloader_num_workers=4 \
  --learning_rate=1e-04 \
  --weight_decay=0.01 \
  --output_dir="/ghome/mpilligua/TFG/IP-Adapter/trainings_xl/run3/" \
  --save_steps=500