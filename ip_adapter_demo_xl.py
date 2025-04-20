# 
# !pip install transformers
# !pip install torch
# !pip install einops
# !pip install accelerate

# 
import torch
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline, StableDiffusionImg2ImgPipeline, StableDiffusionInpaintPipelineLegacy, DDIMScheduler, AutoencoderKL
from PIL import Image

from ip_adapter import IPAdapter, IPAdapterXL

# 
base_model_path = "runwayml/stable-diffusion-v1-5"
base_model_path = "stabilityai/stable-diffusion-xl-base-1.0"
vae_model_path = "stabilityai/sd-vae-ft-mse"
image_encoder_path = "/ghome/mpilligua/TFG/IP-Adapter/models/image_encoder/"
image_encoder_path = "/ghome/mpilligua/TFG/IP-Adapter/sdxl_models/image_encoder/"
ip_ckpt = "/ghome/mpilligua/TFG/IP-Adapter/models/ip-adapter_sd15.bin"
ip_ckpt = "/ghome/mpilligua/TFG/IP-Adapter/trainings_xl/run3/checkpoint-500/ip_adapter2.bin"
device = "cuda"

# 
# from huggingface_hub import snapshot_download
# snapshot_download(repo_id="stabilityai/sd-vae-ft-mse")

# 
def image_grid(imgs, rows, cols):
    assert len(imgs) == rows*cols

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

#  [markdown]
# ## Image Variations

# 
# load SD pipeline
# pipe = StableDiffusionPipeline.from_pretrained(
#     base_model_path,
#     torch_dtype=torch.float16,
#     scheduler=noise_scheduler,
#     vae=vae,
#     feature_extractor=None,
#     safety_checker=None
# )
pipe = StableDiffusionXLPipeline.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16,
    add_watermarker=False,
)

# 
# read image prompt
image = Image.open("/ghome/mpilligua/TFG/Datasets/RSR_256/scene_01/00217_0007_0_0_255_255_255_1_90_1_1.jpg")
image.resize((256, 256))

# # 
# # load ip-adapter
# ip_model = IPAdapter(pipe, image_encoder_path, ip_ckpt, device)
ip_model = IPAdapterXL(pipe, image_encoder_path, ip_ckpt, device)

# # 
# # generate image variations
images = ip_model.generate(pil_image=image, num_samples=4, num_inference_steps=50, seed=42, prompt="a scene with tilt 0.75, and pan 0.19", strength=0.6)
grid = image_grid(images, 1, 4)

# # save the image
grid.save("grid3-xl.png")

#  [markdown]
# ## Image-to-Image

# 
# # load SD Img2Img pipe
# del pipe, ip_model
# torch.cuda.empty_cache()
# pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
#     base_model_path,
#     torch_dtype=torch.float16,
#     scheduler=noise_scheduler,
#     vae=vae,
#     feature_extractor=None,
#     safety_checker=None
# )

# # 
# # read image prompt
# image = Image.open("assets/images/river.png")
# g_image =   Image.open("/ghome/mpilligua/TFG/IP-Adapter/condition.png")
# image_grid([image.resize((256, 256)), g_image.resize((256, 256))], 1, 2)

# # 
# # load ip-adapter
# ip_model = IPAdapter(pipe, image_encoder_path, ip_ckpt, device)

# # # 
# # # generate
# images = ip_model.generate(pil_image=image, num_samples=4, num_inference_steps=50, seed=42, image=g_image, strength=0.6)
# grid = image_grid(images, 1, 4)
# grid.save("grid-im2im-xl.png")

# #  [markdown]
# # ## Inpainting

# # 
# # load SD Inpainting pipe
# del pipe, ip_model
# torch.cuda.empty_cache()
# pipe = StableDiffusionInpaintPipelineLegacy.from_pretrained(
#     base_model_path,
#     torch_dtype=torch.float16,
#     scheduler=noise_scheduler,
#     vae=vae,
#     feature_extractor=None,
#     safety_checker=None
# )

# # 
# # read image prompt
# image = Image.open("assets/images/girl.png")
# image.resize((256, 256))

# # 
# masked_image = Image.open("assets/inpainting/image.png").resize((512, 768))
# mask = Image.open("assets/inpainting/mask.png").resize((512, 768))
# image_grid([masked_image.resize((256, 384)), mask.resize((256, 384))], 1, 2)

# # 
# # load ip-adapter
# ip_model = IPAdapter(pipe, image_encoder_path, ip_ckpt, device)

# # 
# # generate
# images = ip_model.generate(pil_image=image, num_samples=4, num_inference_steps=50,
#                            seed=42, image=masked_image, mask_image=mask, strength=0.7, )
# grid = image_grid(images, 1, 4)
# grid


