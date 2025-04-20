import os
import pandas as pd
from torchvision.io import read_image
import random
from torch.utils.data import Dataset
import torchvision.transforms as T
import torch
from PIL import Image

import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from transformers import CLIPImageProcessor
import cv2
import numpy as np

class CustomDataset(Dataset):
    def __init__(self, tokenizer, size):
        self.image_root_path = "/ghome/mpilligua/TFG/Datasets/RSR_256"
        self.scenes = os.listdir(self.image_root_path)
        self.imgsXscene = {}
        self.imgsXscene = {scene: self.read_imgs_per_scene(scene) for scene in self.scenes}

        self.possible_captions = [
            "a photo with tilt YYYY, and pan ZZZZ",
            "a scene with tilt YYYY, and pan ZZZZ",
            "a composition with tilt YYYY, and pan ZZZZ",
            "an image with tilt YYYY, and pan ZZZZ",
        ]
        
        self.tokenizer = tokenizer
        self.size = size
        self.transform = transforms.Compose([
            transforms.Resize(self.size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(self.size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        self.clip_image_processor = CLIPImageProcessor()

    def read_imgs_per_scene(self, scene_name):
        scene_dir = os.path.join(self.image_root_path, scene_name)
        scene = {}
        # transform = T.ConvertImageDtype(torch.float32)
        for img in os.listdir(scene_dir):
            image = {}
            # img_path = os.path.join(scene_dir, img)
            # image['img'] = transform(read_image(img_path))
            # image['img_name'] = img
            # image['scene'] = scene
            image['pan'], image['tilt'], rgb, image['index_view'] = self.get_pan_and_tilt_from_filename(img)
            
            if rgb != [255, 255, 255]:
                continue
            
            image['img_path'] = os.path.join(scene_name, img)


            if image['index_view'] not in scene:
                scene[image['index_view']] = []
            scene[image['index_view']].append(image)
        return scene

    def get_pan_and_tilt_from_filename(self, filename):
        filename_parts = filename.split('_')
        pan = int(filename_parts[2]) / 360
        tilt = int(filename_parts[3]) / 360
        rgb = [0, 0, 0]  # Placeholder for RGB values
        rgb[0] = int(filename_parts[4])
        rgb[1] = int(filename_parts[5])
        rgb[2] = int(filename_parts[6])
        index_scene = int(filename_parts[1])
        return pan, tilt, rgb, index_scene

    def __len__(self):
        return sum([len(self.imgsXscene[scene][ind]) for scene in self.scenes for ind in self.imgsXscene[scene]])

    def __getitem__(self, idx):
        rand_scene = random.choice(list(self.imgsXscene.keys()))
        rand_index = random.choice(list(self.imgsXscene[rand_scene].keys()))
        rand_img = random.choice(self.imgsXscene[rand_scene][rand_index])

        img1 = rand_img
        img2 = random.choice(self.imgsXscene[rand_scene][rand_index])

        # read image
        gt_image = Image.open(os.path.join(self.image_root_path, img1['img_path']))
        gt_image = self.transform(gt_image.convert("RGB"))
        
        raw_condition_image = Image.open(os.path.join(self.image_root_path, img2['img_path']))
        # condition_image = self.transform(raw_condition_image.convert("RGB"))
        condition_image = self.clip_image_processor(images=raw_condition_image, return_tensors="pt").pixel_values
        
        caption = random.choice(self.possible_captions)
        caption = caption.replace("YYYY", str(img2['tilt'])).replace("ZZZZ", str(img2['pan']))

        # print(caption)

        # get text and tokenize
        text_input_ids = self.tokenizer(
            caption,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).input_ids
        
        print("a", text_input_ids.shape)
        print(caption)
        exit(0)
        return {"gt_image": gt_image,
                "condition_image": condition_image,
                "condition_text": text_input_ids,
                "drop_image_embed": False,
                "input_image": raw_condition_image,
                "path_gt_image": os.path.join(self.image_root_path, img1['img_path']),
                "path_condition_image": os.path.join(self.image_root_path, img2['img_path'])}
        
        
        
def resize_image(input_image, resolution):
    H, W, C = input_image.shape
    H = float(H)
    W = float(W)
    k = float(resolution) / min(H, W)
    H *= k
    W *= k
    H = int(np.round(H / 64.0)) * 64
    W = int(np.round(W / 64.0)) * 64
    img = cv2.resize(input_image, (W, H), interpolation=cv2.INTER_LANCZOS4 if k > 1 else cv2.INTER_AREA)
    return img


def HWC3(x):
    assert x.dtype == np.uint8
    if x.ndim == 2:
        x = x[:, :, None]
    assert x.ndim == 3
    H, W, C = x.shape
    assert C == 1 or C == 3 or C == 4
    if C == 3:
        return x
    if C == 1:
        return np.concatenate([x, x, x], axis=2)
    if C == 4:
        color = x[:, :, 0:3].astype(np.float32)
        alpha = x[:, :, 3:4].astype(np.float32) / 255.0
        y = color * alpha + 255.0 * (1.0 - alpha)
        y = y.clip(0, 255).astype(np.uint8)
        return y


class CannyDetector:
    def __call__(self, img, low_threshold, high_threshold):
        return cv2.Canny(img, low_threshold, high_threshold)

model_canny = CannyDetector()

def canny(img, res=256, l=100, h=200):
    img = resize_image(HWC3(img), res)
    global model_canny
    result = model_canny(img, l, h)
    return [result]
        

class CustomValDataset(Dataset):
    def __init__(self, tokenizer, size):
        self.image_root_path = "/ghome/mpilligua/TFG/Datasets/RSR_256"
        self.scenes = os.listdir(self.image_root_path)
        self.imgsXscene = {}
        self.imgsXscene = {scene: self.read_imgs_per_scene(scene) for scene in self.scenes}

        self.possible_captions = [
            "a photo with tilt YYYY, and pan ZZZZ",
            "a scene with tilt YYYY, and pan ZZZZ",
            "a composition with tilt YYYY, and pan ZZZZ",
            "an image with tilt YYYY, and pan ZZZZ",
        ]
        
        self.tokenizer = tokenizer
        self.size = size
        self.transform = transforms.Compose([
            transforms.Resize(self.size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(self.size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        self.clip_image_processor = CLIPImageProcessor()

    def read_imgs_per_scene(self, scene_name):
        scene_dir = os.path.join(self.image_root_path, scene_name)
        scene = {}
        # transform = T.ConvertImageDtype(torch.float32)
        for img in os.listdir(scene_dir):
            image = {}
            # img_path = os.path.join(scene_dir, img)
            # image['img'] = transform(read_image(img_path))
            # image['img_name'] = img
            # image['scene'] = scene
            image['pan'], image['tilt'], rgb, image['index_view'] = self.get_pan_and_tilt_from_filename(img)
            
            if rgb != [255, 255, 255]:
                continue
            
            image['img_path'] = os.path.join(scene_name, img)


            if image['index_view'] not in scene:
                scene[image['index_view']] = []
            scene[image['index_view']].append(image)
        return scene

    def get_pan_and_tilt_from_filename(self, filename):
        filename_parts = filename.split('_')
        pan = int(filename_parts[2]) / 360
        tilt = int(filename_parts[3]) / 360
        rgb = [0, 0, 0]  # Placeholder for RGB values
        rgb[0] = int(filename_parts[4])
        rgb[1] = int(filename_parts[5])
        rgb[2] = int(filename_parts[6])
        index_scene = int(filename_parts[1])
        return pan, tilt, rgb, index_scene

    def __len__(self):
        return sum([len(self.imgsXscene[scene][ind]) for scene in self.scenes for ind in self.imgsXscene[scene]])

    def __getitem__(self, idx):
        rand_scene = random.choice(list(self.imgsXscene.keys()))
        rand_index = random.choice(list(self.imgsXscene[rand_scene].keys()))
        rand_img = random.choice(self.imgsXscene[rand_scene][rand_index])

        img1 = rand_img
        img2 = random.choice(self.imgsXscene[rand_scene][rand_index])

        # read image
        gt_image = Image.open(os.path.join(self.image_root_path, img1['img_path']))
        # gt_image = self.transform(gt_image.convert("RGB"))
        
        raw_condition_image = Image.open(os.path.join(self.image_root_path, img2['img_path']))
        # condition_image = self.transform(condition_image.convert("RGB"))
        condition_image = self.clip_image_processor(images=raw_condition_image, return_tensors="pt").pixel_values
        
        numpy_condition_image = cv2.imread(os.path.join(self.image_root_path, img2['img_path']))
        canny_edges = canny(numpy_condition_image, res=256, l=125, h=150)
        
        caption = random.choice(self.possible_captions)
        caption = caption.replace("YYYY", str(img2['tilt'])).replace("ZZZZ", str(img2['pan']))
        print(caption)

        # get text and tokenize
        text_input_ids = self.tokenizer(
            caption,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).input_ids
        print(text_input_ids.shape)

        return {"gt_image": gt_image,
                "condition_image": condition_image,
                "condition_text": text_input_ids,
                "drop_image_embed": False,
                "input_image": raw_condition_image,
                "path_gt_image": os.path.join(self.image_root_path, img1['img_path']),
                "path_condition_image": os.path.join(self.image_root_path, img2['img_path']),
                "canny_edges": canny_edges}