# video generation
import os
import sys
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
sys.path.append("./DiffSynth-Studio")
from diffsynth import ModelManager, WanVideoPipeline
from utils_3dscene.nvrender import perform_camera_movement_with_cam_input, load_rail
from utils_3dscene.pipeline_utils_3dscene import write_video
from PIL import Image
import imageio
import argparse
import numpy as np
import torch
import cv2
from torchvision.transforms import v2
from einops import rearrange
import torchvision
import torch.distributed as dist
from modelscope import snapshot_download
from xfuser.core.distributed import init_distributed_environment, initialize_model_parallel
from pathlib import Path
import json
# add 5b model support;
from diffsynth.pipelines.wan_video_new import WanVideoPipelineNew, ModelConfig
from diffsynth.trainers.utils import VideoDataset
from peft import LoraConfig, inject_adapter_in_model
from diffsynth import load_state_dict
def add_lora_to_model(model, target_modules, lora_rank, lora_alpha=None):
    if lora_alpha is None:
        lora_alpha = lora_rank
    lora_config = LoraConfig(r=lora_rank, lora_alpha=lora_alpha, target_modules=target_modules)
    model = inject_adapter_in_model(lora_config, model)
    return model
MASK_RATIO = 0.
class TextVideoDataset(torch.utils.data.Dataset):
    def __init__(self, vid_path, mask_path,text, max_num_frames=81, frame_interval=1, num_frames=81, height=720, width=1440, is_i2v=True):

        self.path = [vid_path]
        self.mask_video_path = [mask_path]
        self.text = [text]
        
        self.max_num_frames = max_num_frames
        self.frame_interval = frame_interval
        self.num_frames = num_frames
        self.height = height
        self.width = width
        self.is_i2v = is_i2v
        
        # this should not be center crop
        # should be 
        self.frame_process = v2.Compose([
            #v2.CenterCrop(size=(height, width)),
            v2.Resize(size=(height, width), antialias=True),
            v2.ToTensor(),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        
        
    def crop_and_resize(self, image):
        width, height = image.size
        scale = max(self.width / width, self.height / height)
        image = torchvision.transforms.functional.resize(
            image,
            (round(height*scale), round(width*scale)),
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR
        )
        return image
    
    def crop_and_resize_standard(self, image):
        width, height = image.size
        #scale = max(self.width / width, self.height / height)
        image = torchvision.transforms.functional.resize(
            image,
            (self.width, self.height),
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR
        )
        return image


    def load_frames_using_imageio(self, file_path, max_num_frames, start_frame_id, interval, num_frames, frame_process):
        reader = imageio.get_reader(file_path)
        if reader.count_frames() < max_num_frames or reader.count_frames() - 1 < start_frame_id + (num_frames - 1) * interval:
            reader.close()
            return None
        
        frames = []
        first_frame = None
        for frame_id in range(num_frames):
            frame = reader.get_data(start_frame_id + frame_id * interval)
            frame = Image.fromarray(frame)
            frame = self.crop_and_resize(frame)
            if first_frame is None:
                first_frame = frame
            frame = frame_process(frame)
            frames.append(frame)
        reader.close()

        frames = torch.stack(frames, dim=0)
        frames = rearrange(frames, "T C H W -> C T H W")
        
        first_frame = v2.functional.center_crop(first_frame, output_size=(self.height, self.width))
        first_frame = np.array(first_frame)

        if self.is_i2v:
            return frames, first_frame
        else:
            return frames

    
    
    
    

    def load_frames_using_imageio_standard(self, file_path, max_num_frames, start_frame_id, interval, num_frames, frame_process):
        reader = imageio.get_reader(file_path)
        if reader.count_frames() < max_num_frames or reader.count_frames() - 1 < start_frame_id + (num_frames - 1) * interval:
            reader.close()
            return None
        
        frames = []
        first_frame = None
        for frame_id in range(num_frames):
            frame = reader.get_data(start_frame_id + frame_id * interval)
            frame = Image.fromarray(frame)
            frame = self.crop_and_resize_standard(frame)
            if first_frame is None:
                first_frame = frame
            frame = frame_process(frame)
            frames.append(frame)
        reader.close()

        frames = torch.stack(frames, dim=0)
        frames = rearrange(frames, "T C H W -> C T H W")
        
        first_frame = v2.functional.center_crop(first_frame, output_size=(self.height, self.width))
        first_frame = np.array(first_frame)


        return frames


    def load_video(self, file_path):
        start_frame_id = torch.randint(0, 1, (1,))[0]
        frames = self.load_frames_using_imageio_standard(file_path, self.max_num_frames, start_frame_id, self.frame_interval, self.num_frames, self.frame_process)
        return frames
    
    
    def is_image(self, file_path):
        file_ext_name = file_path.split(".")[-1]
        if file_ext_name.lower() in ["jpg", "jpeg", "png", "webp"]:
            return True
        return False
    
    
    def load_image(self, file_path):
        frame = Image.open(file_path).convert("RGB")
        frame = self.crop_and_resize(frame)
        first_frame = frame
        frame = self.frame_process(frame)
        frame = rearrange(frame, "C H W -> C 1 H W")
        return frame


    def __getitem__(self, data_id):
        text = self.text[data_id]
        path = self.path[data_id]
        mask_video_path = self.mask_video_path[data_id]


        video = self.load_video(path)
        mask_video = self.load_video(mask_video_path)
        #print(video.max(), video.min(), mask_video.max(), mask_video.min())
        mask_bool = mask_video < MASK_RATIO
        masked_video = video.clone()
        masked_video[mask_bool] = -1.
        data = {"text": text, "video": video, "path": path, "masked_video": masked_video, "mask_video":mask_video}
        # HACK: save video for visualize

        return data
    

    def __len__(self):
        return len(self.path)
training_iters=3000 # optimization iterations
num_of_point_cloud=3000000 # number of point cloud unprojected from depth map
num_views_per_view=3 # inserted between adjacent camera poses
img_sample_interval=1 # images selected during training to optimize 3DGS
moge_ckpt_path = os.path.abspath("checkpoints/moge/model.pt")

def simple_filename(prompt):
    filename = re.sub(r'[^\w\s-]', '', prompt)  
    filename = re.sub(r'\s+', '_', filename)    
    return f"{filename[:50]}"              


def main(args):
    output_dir = args.inout_dir
    panorama_path = os.path.join(output_dir, 'pano_img.jpg') if os.path.exists(os.path.join(output_dir, 'pano_img.jpg')) else os.path.join(output_dir, 'pano_img.png')
    prompt_path = os.path.join(output_dir,"prompt.txt")
    # whether to use 5b model for video generation;
    use_5b_model = args.use_5b_model


    dist.init_process_group(
        backend="nccl",
        init_method="env://",
    )
    # Download models
    
    from xfuser.core.distributed import (initialize_model_parallel,
                                        init_distributed_environment)
    init_distributed_environment(
        rank=dist.get_rank(), world_size=dist.get_world_size())

    initialize_model_parallel(
        sequence_parallel_degree=dist.get_world_size(),
        ring_degree=1,
        ulysses_degree=dist.get_world_size(),
    )

    torch.cuda.set_device(dist.get_rank())
    
    print(f"\n\n{panorama_path}\n\n")

    
    with open(prompt_path,"r",encoding="utf-8") as f:
        prompt=f.read()
        print(f"prompt is {prompt}")
    angle = args.angle
    movement_range = args.movement_range
    movement_mode = args.movement_mode
    seed = args.seed
    resolution = args.resolution
    is_720p = resolution == 720

    if is_720p:
        if not use_5b_model:
            snapshot_download("Wan-AI/Wan2.1-I2V-14B-720P", local_dir="checkpoints/Wan-AI/Wan2.1-I2V-14B-720P")
        # download model only when not using the 5b model;
    else:
        snapshot_download("Wan-AI/Wan2.1-I2V-14B-480P", local_dir="checkpoints/Wan-AI/Wan2.1-I2V-14B-480P")
    # do other things only in the main rank;
    device = f"cuda:{dist.get_rank()}"
    case_dir = os.path.abspath(output_dir)#os.path.abspath(os.path.join(output_dir, panorama_name))
    os.makedirs(case_dir,exist_ok=True)
    print(f"panorama_path={panorama_path}")
    panorama = cv2.resize(cv2.imread(panorama_path, cv2.IMREAD_UNCHANGED),(2048,1024),interpolation=cv2.INTER_AREA)
    input_image_path = os.path.join(case_dir, "moge.png")
    cv2.imwrite(input_image_path, panorama)
    if dist.get_rank() == 0:
        print("\n\nperform moge...\n\n")
        os.system(f"cd code/MoGe && python scripts/infer_panorama.py --input {os.path.abspath(input_image_path)} --output {case_dir} --pretrained {moge_ckpt_path} --device {device} --threshold 0.03 --maps --ply")
        depth_path = os.path.join(case_dir, "moge","depth.exr")
        mask_path = os.path.join(case_dir, "moge", "mask.png")
        print(f"{os.path.exists(mask_path)},{mask_path}")
        depth = cv2.imread(depth_path, cv2.IMREAD_ANYCOLOR|cv2.IMREAD_ANYDEPTH)
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)[:,:] > 127
        valid_max = depth[mask].max()
        depth[~mask] = 2. * valid_max

        panorama_torch = (torch.from_numpy(panorama).float()/255.).to("cuda")
        depth_torch = torch.from_numpy(depth).float().to("cuda")
        mask_torch = torch.from_numpy(mask).bool().to("cuda")
        
        if len(args.json_path) > 0 and os.path.exists(args.json_path):
            rail = load_rail(args.json_path)
        else:
            rail = None
        rendered_rgb, rendered_mask, render_Rts, firstframe_rgb, firstframe_depth, angle = perform_camera_movement_with_cam_input(panorama_torch, depth_torch, angle=angle, movement_ratio=movement_range, frame_size=81, preset_rail=rail,mode=movement_mode)

    condition_dir = os.path.join(case_dir,"condition")
    os.makedirs(condition_dir, exist_ok=True)
    camera_path = os.path.join(condition_dir, "cameras.npz")
    if dist.get_rank() == 0:
        rendered_rgb_np = (rendered_rgb.cpu().numpy() * 255.).astype(np.uint8)
        rendered_mask_np = (rendered_mask.float()[:,:,:,None].repeat(1,1,1,3).cpu().numpy() * 255.).astype(np.uint8)

        write_video(rendered_rgb_np, os.path.join(condition_dir,"rendered_rgb.mp4"), 12)
        write_video(rendered_mask_np, os.path.join(condition_dir,"rendered_mask.mp4"), 12)

        W = firstframe_rgb.shape[1]
        q = int(angle/360. * W + W//2)%W
        mask_torch = torch.cat([mask_torch[:,q:],mask_torch[:,:q]],dim=1)
        mask_torch = torch.cat([mask_torch[:,W//2:],mask_torch[:,:W//2]],dim=1)

        cv2.imwrite(os.path.join(condition_dir,"firstframe_rgb.png"), (firstframe_rgb.cpu().numpy()*255.).astype(np.uint8))
        cv2.imwrite(os.path.join(condition_dir,"firstframe_depth.exr"), (firstframe_depth.cpu().numpy()))
        cv2.imwrite(os.path.join(condition_dir,"firstframe_mask.png"), mask_torch.cpu().numpy().astype(np.uint8)*255)
        
    
        np.savez(camera_path, render_Rts.cpu().numpy())
        

    # perform panovid generation;
    dist.barrier()
    if not use_5b_model:
        model_manager = ModelManager(torch_dtype=torch.bfloat16, device="cpu")
        if is_720p:
            model_manager.load_models(
                ["./checkpoints/Wan-AI/Wan2.1-I2V-14B-720P/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"],
                torch_dtype=torch.float32, # Image Encoder is loaded with float32
            ) 
            BASE_DIR = Path(__file__).parent.absolute()
            BASE_DIR=BASE_DIR.parent

            model_manager.load_models([
                [str(BASE_DIR / f"checkpoints/Wan-AI/Wan2.1-I2V-14B-720P/diffusion_pytorch_model-0000{i}-of-00007.safetensors") for i in range(1, 8)],
                str(BASE_DIR / "checkpoints/Wan-AI/Wan2.1-I2V-14B-720P/models_t5_umt5-xxl-enc-bf16.pth"),
                str(BASE_DIR / "checkpoints/Wan-AI/Wan2.1-I2V-14B-720P/Wan2.1_VAE.pth")
            ])

            model_manager.load_lora("./checkpoints/Wan-AI/wan_lora/pano_video_gen_720p.bin", lora_alpha=1.0)
        else:
            model_manager.load_models(
                ["./checkpoints/Wan-AI/Wan2.1-I2V-14B-480P/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"],
                torch_dtype=torch.float32, # Image Encoder is loaded with float32
            ) 
            BASE_DIR = Path(__file__).parent.absolute()
            BASE_DIR=BASE_DIR.parent

            model_manager.load_models([
                [str(BASE_DIR / f"checkpoints/Wan-AI/Wan2.1-I2V-14B-480P/diffusion_pytorch_model-0000{i}-of-00007.safetensors") for i in range(1, 8)],
                str(BASE_DIR / "checkpoints/Wan-AI/Wan2.1-I2V-14B-480P/models_t5_umt5-xxl-enc-bf16.pth"),
                str(BASE_DIR / "checkpoints/Wan-AI/Wan2.1-I2V-14B-480P/Wan2.1_VAE.pth")
            ])

            model_manager.load_lora("./checkpoints/Wan-AI/wan_lora/pano_video_gen_480p.ckpt", lora_alpha=1.0)

        pipe = WanVideoPipeline.from_model_manager(model_manager, device=f"cuda:{dist.get_rank()}",use_usp=True if dist.get_world_size() > 1 else False)
        if args.enable_vram_management:
            pipe.enable_vram_management(num_persistent_param_in_dit=0)
        else:
            pipe.enable_vram_management(num_persistent_param_in_dit=None)
    else:
        pipe = WanVideoPipelineNew.from_pretrained(
            torch_dtype=torch.bfloat16,
            device="cuda",
            model_configs=[
                ModelConfig(local_model_path="./checkpoints", model_id="Wan-AI/Wan2.2-TI2V-5B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth", offload_device="cpu"),
                ModelConfig(local_model_path="./checkpoints", model_id="Wan-AI/Wan2.2-TI2V-5B", origin_file_pattern="diffusion_pytorch_model*.safetensors", offload_device="cpu"),
                ModelConfig(local_model_path="./checkpoints", model_id="Wan-AI/Wan2.2-TI2V-5B", origin_file_pattern="Wan2.2_VAE.pth", offload_device="cpu"),
            ],
            use_usp=True if dist.get_world_size() > 1 else False
        )
        lora_checkpoint = os.path.abspath("./checkpoints/Wan-AI/wan-lora/pano_video_gen_720p_5b.safetensors")
        model = add_lora_to_model(
            getattr(pipe, "dit"),
            "q,k,v,o,ffn.0,ffn.2".split(","),
            48,
            48.0
        )
        state_dict = load_state_dict(lora_checkpoint)
        load_result = model.load_state_dict(state_dict, strict=False)

        if args.enable_vram_management:
            pipe.enable_vram_management(num_persistent_param_in_dit=0)
        else:
            pipe.enable_vram_management(num_persistent_param_in_dit=None)


    if not use_5b_model:
        #vid_path, mask_path,text,
        tgt_resolution = (960,720) if is_720p else (960,480)
        #dset = TextVideoDataset(vid_path = os.path.join(condition_dir,"rendered_rgb.mp4"), mask_path = os.path.join(condition_dir,"rendered_mask.mp4"), text=prompt)
        # (self, vid_path, mask_path,text, max_num_frames=81, frame_interval=1, num_frames=81, height=720, width=1440, is_i2v=True):
        dset = TextVideoDataset(vid_path = os.path.join(condition_dir,"rendered_rgb.mp4"), mask_path = os.path.join(condition_dir,"rendered_mask.mp4"), text=prompt, height=tgt_resolution[1],width=tgt_resolution[0])
        cases = dset[0]
        prompt = cases["text"]
        cond_video = ((cases["masked_video"].permute(1,2,3,0) + 1.) / 2. * 255.).cpu().numpy()
        cond_mask = ((cases["mask_video"].permute(1,2,3,0) + 1.) / 2. * 255.).cpu().numpy()
        #print(prompt[i])
        video = pipe(
            prompt=prompt+" The video is of high quality, and the view is very clear. High quality, masterpiece, best quality, highres, ultra-detailed, fantastic.",
            negative_prompt="The video is not of a high quality, it has a low resolution. Distortion. strange artifacts.",
            cfg_scale=5.0,
            num_frames=161,
            num_inference_steps=5,
            seed=seed, tiled=True,
            height=tgt_resolution[1],
            width=tgt_resolution[0],
            cond_video = cond_video,
            cond_mask = cond_mask
        )
    else:
        tgt_resolution = (1408,704)
        height = tgt_resolution[1]
        width = tgt_resolution[0]
        # TODO: add no-csv input support;
        condition_info = [
            {
                "video":os.path.join(condition_dir,"rendered_rgb.mp4"),
                "cond_video":os.path.join(condition_dir,"rendered_rgb.mp4"),
                "cond_mask":os.path.join(condition_dir,"rendered_mask.mp4"),
                "prompt":prompt
            }
        ]
        dset = VideoDataset(
            #base_path="/", metadata_path="/datasets_3d/zhongqi.yang/matrix3d_inference/dataset/metadata_1k.csv",
            base_path="/", metadata_path=None,
            num_frames=161,
            time_division_factor=4, time_division_remainder=1,
            max_pixels=height*width, height=height, width=width,
            height_division_factor=16, width_division_factor=16,
            data_file_keys=("video","cond_video","cond_mask"),
            image_file_extension=("jpg", "jpeg", "png", "webp"),
            video_file_extension=("mp4", "avi", "mov", "wmv", "mkv", "flv", "webm"),
            repeat=1,
            args=None,
            raw_dict=condition_info
        )
        cases = dset[0]

        video_ori = pipe(
            prompt=cases['prompt'] + " The video is of high quality, and the view is very clear. High quality, masterpiece, best quality, highres, ultra-detailed, fantastic.",
            negative_prompt="The video is not of a high quality, it has a low resolution. Distortion. strange artifacts. flickering. worst quality. low quality",
            seed=120, tiled=True,
            height=height, width=width,
            input_image=cases["video"][0],
            num_frames=161,
            cond_video = (cases["cond_video"]),
            cond_mask = (cases["cond_mask"]),
        )
        # the original resolution of 5b model is actually [704,1408], in order to be unified with latter steps, we resize the output to [960,720].
        video = [img.resize((960,720)) for img in video_ori]

    if dist.get_rank() == 0:
        generated_dir = os.path.join(case_dir,"generated")
        generated_path = os.path.join(generated_dir,"generated.mp4")
        
        os.makedirs(generated_dir, exist_ok=True)
        result = []
        for j in range(81):
            generated_image = np.array(video[j])[:,:,::-1]
            result.append(generated_image)
        write_video(result, generated_path, 24)
    
        # gather output;
        gathered_video_name = f"pano_video.mp4"
        all_output_dir = case_dir
        os.makedirs(all_output_dir, exist_ok=True)
        os.system(f"cp {generated_path} {os.path.join(all_output_dir, gathered_video_name)}")
        all_cameras_list = render_Rts.cpu().numpy().tolist()

        pano_camera_path = os.path.join(all_output_dir, "pano_video_cam.json")
        with open(pano_camera_path, "w") as F_:
            F_.write(json.dumps(all_cameras_list,indent=4))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--angle", type=float, default=0., help="the azimuth angle of camera movement direction. angle=0 means the camera moves towards the center of the panoramic image, angle=90 means the camera moves towards the middle-right direction of the panoramic image")
    parser.add_argument("--movement_range", type=float, default=0.6, help="relative movement range of the camera w.r.t the estimated depth of the input panorama. the value should be between 0~0.8")
    parser.add_argument("--movement_mode", type=str, default="straight", help="the shape of the rail along which the camera moves. choose between ['s_curve','l_curve','r_curve','straight']")
    parser.add_argument("--json_path", type=str, default="", help="predefined camera path. the predefined camera is stored as json file in the format defined in code/generate_example_camera.py")#######2025-6-13
    parser.add_argument("--seed", type=int, default=0, help="the generation seed")
    parser.add_argument("--resolution", type=int, default=720, help="the working resolution of the panoramic video generation model.")
    parser.add_argument("--inout_dir", type=str, default="./output/example1")
    parser.add_argument("--use_5b_model", action="store_true", help="whether to use 5b model to train things.")
    parser.add_argument("--enable_vram_management", action="store_true", help="whether to enable vram management for running on low mem devices.")
    args = parser.parse_args()
    main(args)
