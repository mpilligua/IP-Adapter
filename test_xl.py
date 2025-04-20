import torch
from safetensors.torch import load_file

path = "/ghome/mpilligua/TFG/IP-Adapter/trainings_xl/run3/checkpoint-500"
ckpt = f"{path}/model.safetensors"
sd = load_file(ckpt)

image_proj_sd = {}
ip_sd = {}
countIpK = 1
countIpV = 1

for k in sd:
    if k.startswith("image_proj_model"):
        image_proj_sd[k.replace("image_proj_model.", "")] = sd[k]

for k in sd:
    if "down_" in k:
        if "to_k_ip." in k:
            ip_sd[str(countIpK) + '.to_k_ip.weight'] = sd[k]
            countIpK += 2
        elif "to_v_ip." in k:
            ip_sd[str(countIpV) + '.to_v_ip.weight'] = sd[k]
            countIpV += 2

for k in sd:
    if "mid_" in k:
        if "to_k_ip." in k:
            ip_sd[str(countIpK) + '.to_k_ip.weight'] = sd[k]
            countIpK += 2
        elif "to_v_ip." in k:
            ip_sd[str(countIpV) + '.to_v_ip.weight'] = sd[k]
            countIpV += 2

for k in sd:
    if "up_" in k:
        if "to_k_ip." in k:
            ip_sd[str(countIpK) + '.to_k_ip.weight'] = sd[k]
            countIpK += 2
        elif "to_v_ip." in k:
            ip_sd[str(countIpV) + '.to_v_ip.weight'] = sd[k]
            countIpV += 2

torch.save({"image_proj": image_proj_sd, "ip_adapter": ip_sd}, f"{path}/ip_adapter_up_mid_down_sdxl.bin")