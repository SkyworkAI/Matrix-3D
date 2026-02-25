from pano_init.src.worldgen import WorldGen
from MoGe.moge.model.moge_model import MoGeModel
from pano_init.prompt.prompt import Lamma_Video
# from pano_init.prompt.intervl import InterVL
import torch
from torch import device
from PIL import Image, ImageOps
import os
import json
import cv2
import math
from huggingface_hub import hf_hub_download



class i2pano:
    def __init__(self, device):


        self.device = device
        model_path = "checkpoints/moge/model.pt"
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        if not os.path.exists(model_path):
            hf_hub_download(
                repo_id="Ruicheng/moge-vitl",
                filename="model.pt",
                local_dir=os.path.dirname(model_path),
                local_dir_use_symlinks=False,
                force_download=True
            )
        self.moge = MoGeModel.from_pretrained(model_path)
        self.moge = self.moge.to(self.device)
        self.worldgen=WorldGen(mode="i2s", device=self.device)
        self.Lamma_Video = Lamma_Video(self.device)

        
    def inpaint_img(self, img_path,seed=42,prompt=None,fov=None):
        if fov is None:
            hFov, wFov= self.calculate_FOV(img_path) 
        else:
            hFov = fov
            wFov = fov
        try:
            if prompt is not None:
                prompt = prompt
            else:
                prompt = self.Lamma_Video.extract_prompt(img_path,debug=True)
        except Exception:
            print("Lamma Video prompt failed")
            promt="a lot of trees"
        print(f"Lamma Video prompt {prompt}")
        prompt_copy=prompt
        pano_img = self.worldgen.generate(img_path, prompt, 
                                          wFov, hFov,seed)
        return pano_img,prompt_copy

    def calculate_FOV(self, img_path, debug=True):
        input_image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        input_image = torch.tensor(input_image / 255, dtype=torch.float32, device=self.device).permute(2, 0, 1)   
        #infer
        output = self.moge.infer(input_image)
        K =output["intrinsics"]
        def calculateFov(focal, size):
            return 2*math.degrees(math.atan(size/(2*focal)))   
        fx, fy, cx, cy = K[0, 0].item(), K[1, 1].item(), K[0, 2].item(), K[1, 2].item()
        img_h = 2*cy
        img_w = 2*cx
        hFov = calculateFov(fy, img_h)
        wFov = calculateFov(fx, img_w)
        if debug:
            print(f"hFov={hFov}, wFov={wFov}")
        return hFov, wFov         

if __name__ == "__main__":     
    device = torch.device("cuda")
    model = i2pano(device)
    img_path = "/ai-video-sh/haoyuan.li/AIGC/Panodiff/datasets/ours/split_mp4/pers/Rail_00_22_Take_051_rgb.jpg"
    pano_img = model.inpaint_img(img_path)
    pano_img.save("/ai-video-sh/haoyuan.li/AIGC/WorldGen/debug_img/test_pano.jpg")









