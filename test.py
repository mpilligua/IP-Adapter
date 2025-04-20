import torch
path = "/ghome/mpilligua/TFG/IP-Adapter/trainings/2025-04-19_14-23-17/checkpoint-300"
ckpt = f"{path}/pytorch_model.bin"
sd = torch.load(ckpt, map_location="cpu")
image_proj_sd = {}
ip_sd = {}
for k in sd:
    if k.startswith("unet"):
        pass
    elif k.startswith("image_proj_model"):
        image_proj_sd[k.replace("image_proj_model.", "")] = sd[k]
    elif k.startswith("adapter_modules"):
        ip_sd[k.replace("adapter_modules.", "")] = sd[k]

torch.save({"image_proj": image_proj_sd, "ip_adapter": ip_sd}, f"{path}/ip_adapter.bin")

# import torch
# from safetensors.torch import load_file

# ckpt = "/ghome/mpilligua/TFG/IP-Adapter/trainings_xl/run3/checkpoint-500/model.safetensors"
# sd = torch.load(ckpt, map_location="cpu")
# sd = load_file(ckpt, device="cpu")
# image_proj_sd = {}
# ip_sd = {}
# for k in sd:
#     if k.startswith("unet"):
#         pass
#     elif k.startswith("image_proj_model"):
#         image_proj_sd[k.replace("image_proj_model.", "")] = sd[k]
#     elif k.startswith("adapter_modules"):
#         ip_sd[k.replace("adapter_modules.", "")] = sd[k]

# torch.save({"image_proj": image_proj_sd, "ip_adapter": ip_sd}, "/ghome/mpilligua/TFG/IP-Adapter/trainings_xl/run3/checkpoint-500/ip_adapter2.bin")