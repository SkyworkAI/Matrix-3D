import argparse
import torch
import os
from PIL import Image 
import numpy as np
import random
import re
from pano_init.utils.pipeline_flux import FluxPipeline
from pano_init.src.worldgen.models.flux_pano_fill_pipeline import FluxFillPipeline
from pano_init.i2p_model import i2pano

def create_output_dir(base_path: str, prefix: str = "example") -> str:
    os.makedirs(base_path, exist_ok=True)
    max_num = 0
    for dirname in os.listdir(base_path):
        match = re.match(f"{prefix}(\\d+)", dirname)
        if match:
            max_num = max(max_num, int(match.group(1)))
    new_dir = f"{prefix}{max_num + 1}"
    full_path = os.path.join(base_path, new_dir)
    os.makedirs(full_path)
    return full_path
def simple_filename(prompt):
    filename = re.sub(r'[^\w\s-]', '', prompt)  
    filename = re.sub(r'\s+', '_', filename)    
    return f"{filename[:50]}"              

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

def main(args):
    # import pdb
    # pdb.set_trace()
    device=args.device
    seed=args.seed
    setup_seed(seed)
    if(args.mode=="t2p"):
        #t2p_Pipeline = FluxFillPipeline.from_pretrained("black-forest-labs/FLUX.1-Fill-dev", torch_dtype=torch.bfloat16).to(device)
        # flux_path="/mnt/datasets_3d/common/huggingface/hub/models--black-forest-labs--FLUX.1-dev/snapshots/0ef5fff789c832c5c7f4e127f94c8b54bbcced44"
        # t2p_Pipeline=FluxPipeline.from_pretrained(flux_path, torch_dtype=torch.bfloat16).to(device)
        t2p_Pipeline = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev",
            torch_dtype=torch.bfloat16,
        ).to(device) 
        
        lora_path="./checkpoints/flux_lora/pano_image_lora.safetensors"
        t2p_Pipeline.load_lora_weights(lora_path)
        t2p_Pipeline.enable_model_cpu_offload()
        t2p_Pipeline.enable_vae_tiling()
        prompt = args.prompt
        print(f"input prompt={prompt}")
        pano_image = t2p_Pipeline(prompt, 
                height=512,
                width=1024,
                generator=torch.Generator("cpu").manual_seed(seed),
                num_inference_steps=50, 
                blend_extend=0,
                guidance_scale=7).images[0] 
        # output_dir = create_output_dir(args.output_path)
        # output_dir = os.path.join(args.output_path,simple_filename(args.prompt))
        output_dir = args.output_path
    if(args.mode=="i2p"):
        i2p_Pipeline = i2pano(device)
        pano_image, prompt =i2p_Pipeline.inpaint_img(args.input_image_path,seed,args.prompt, args.fov)
        print(f"prompt generated is {prompt}")
        # output_dir = create_output_dir(args.output_path)
        # img_name = os.path.basename(args.input_image_path).split(".")[0]
        # output_dir = os.path.join(args.output_path, img_name)
        output_dir = args.output_path

    

    os.makedirs(output_dir, exist_ok=True) 
    pano_path = os.path.join(output_dir, 'pano_img.jpg')
    pano_image.save(pano_path)
    
    prompt_path = os.path.join(output_dir,"prompt.txt")
    with open(prompt_path, "w", encoding="utf-8") as f:
        f.write(prompt)
    
    # gather results
    '''all_output_dir = os.path.join(args.output_path, "pano_img","all_output")
    os.makedirs(all_output_dir, exist_ok=True)
    os.system(f"cp {prompt_path} {os.path.join(all_output_dir, 'prompt_panoramic_video.txt')}")
    os.system(f"cp {pano_path} {os.path.join(all_output_dir, 'panoraic_image.jpg')}")
    '''


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", 
                   type=str, 
                   default="t2p",
                   choices=["i2p", "t2p"],
                   help="Panorama generation mode selection: "
                        "i2p (image-to-panorama) or t2p (text-to-panorama)")
    parser.add_argument("--prompt", 
                   type=str, 
                   default=None,
                   help="Textual description for panorama generation. "
                        "Required when mode=t2p (e.g. 'A sunset over mountain range')")
    parser.add_argument("--input_image_path", 
                   type=str, 
                   default="data/aquatic_landscape__001.jpg",
                   help="Source image path for i2p mode. "
                        "Supports JPEG/PNG formats")

    parser.add_argument("--device", type=str, default="cuda:0",help="main device")
    parser.add_argument("--seed", type=int, default=380)
    parser.add_argument("--fov", type=float, default=None)
    parser.add_argument("--output_path", type=str, default="",help="Directory path for saving results. ")

    args = parser.parse_args()
    main(args)


