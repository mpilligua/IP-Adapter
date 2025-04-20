import cv2
import numpy as np
import time
import torch
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline, StableDiffusionImg2ImgPipeline, StableDiffusionInpaintPipelineLegacy, DDIMScheduler, AutoencoderKL, StableDiffusionControlNetPipeline, ControlNetModel
from PIL import Image

from ip_adapter import IPAdapter, IPAdapterXL
from dataset import *
import os

# 
base_model_path = "runwayml/stable-diffusion-v1-5"
vae_model_path = "stabilityai/sd-vae-ft-mse"
image_encoder_path = "/ghome/mpilligua/TFG/IP-Adapter/models/image_encoder/"

# load controlnet
controlnet_model_path = "lllyasviel/control_v11f1p_sd15_depth"

run = "/ghome/mpilligua/TFG/IP-Adapter/trainings/run5/"
ckpt = "checkpoint-110"
ip_ckpt = f"{run}/{ckpt}/ip_adapter.bin"

path_save_imgs = f"{run}/imgs-{ckpt}-controlnet/"
os.makedirs(path_save_imgs, exist_ok=True)

device = "cuda"

# 
def image_grid(imgs, rows, cols):
    assert len(imgs) == rows*cols

    imgs = [img.resize((512, 512)) for img in imgs]

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size
    
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid

noise_scheduler = DDIMScheduler(
    num_train_timesteps=1000,
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear",
    clip_sample=False,
    set_alpha_to_one=False,
    steps_offset=1,
)
vae = AutoencoderKL.from_pretrained(vae_model_path).to(dtype=torch.float16)


controlnet = ControlNetModel.from_pretrained(controlnet_model_path, torch_dtype=torch.float16)
# load SD pipeline
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    base_model_path,
    controlnet=controlnet,
    torch_dtype=torch.float16,
    scheduler=noise_scheduler,
    vae=vae,
    feature_extractor=None,
    safety_checker=None
)

def collate_fn(data):
    # gt_images = torch.stack([example["gt_image"] for example in data])
    text_input_ids = torch.cat([example["condition_text"] for example in data], dim=0)
    clip_images = torch.cat([example["condition_image"] for example in data], dim=0)
    drop_image_embeds = [example["drop_image_embed"] for example in data]
    gt_images = [example["gt_image"] for example in data]
    raw_condition_images = [example["input_image"] for example in data]
    path_gt_images = [example["path_gt_image"] for example in data]
    path_condition_images = [example["path_condition_image"] for example in data]
    canny_edges = [example["canny_edges"] for example in data]

    return {"gt_images": gt_images,
                "condition_images": clip_images,
                "condition_texts": text_input_ids,
                "drop_image_embeds": drop_image_embeds,
                "input_images": raw_condition_images,
                "path_gt_images": path_gt_images,
                "path_condition_images": path_condition_images,
                "canny_edges": canny_edges}

batch_size = 2
tokenizer = pipe.tokenizer
eval_dataset = CustomValDataset(size=512, tokenizer=tokenizer)
eval_dataloader = torch.utils.data.DataLoader(
    eval_dataset,
    collate_fn=collate_fn,
    batch_size=batch_size,
    num_workers=4,
)

ip_model = IPAdapter(pipe, image_encoder_path, ip_ckpt, device)



for i, batch in enumerate(eval_dataloader):
    # extract the canny edges of an image


    images = ip_model.generate(pil_image=batch["input_images"], 
                               text_input_ids=batch["condition_texts"],
                               image=batch["canny_edges"],
                               num_samples=3, num_inference_steps=50, 
                               seed=42, strength=0.6)
    
    images = [batch["input_images"][0]] + images + [batch["gt_images"][0]]
    grid = image_grid(images, 1, 5)

    # # save the image
    grid.save(os.path.join(path_save_imgs, f"grid-{i}.jpg"))
