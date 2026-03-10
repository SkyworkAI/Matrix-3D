import argparse
import os
# os.environ["HF_ENDPOINT"] = 'https://hf-mirror.com'
from huggingface_hub import hf_hub_download, snapshot_download, login

# Set HF_TOKEN environment variable before running this script, e.g.:
# export HF_TOKEN=hf_...

def download_ckpt(local_dir, repo_id, filename):
    os.makedirs(local_dir, exist_ok=True)
    local_path = os.path.join(local_dir, os.path.basename(filename))
    if not os.path.exists(local_path):
        file_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=local_dir,
        )
        print(f"File has been downloaded to: {file_path}")
    else:
        print(f"File exists already: {local_path}")


def download_snapshot(local_dir, repo_id):
    if os.path.isdir(local_dir) and os.listdir(local_dir):
        print(f"Already exists: {local_dir}")
        return
    os.makedirs(local_dir, exist_ok=True)
    print(f"\nDownloading snapshot {repo_id} → {local_dir}...\n")
    snapshot_download(repo_id, local_dir=local_dir)
    print(f"Done: {local_dir}")


parser = argparse.ArgumentParser()
parser.add_argument("--resolution", choices=["720p", "480p", "both"], default="720p",
                    help="Which Wan2.1-I2V model to download (720p, 480p, or both)")
parser.add_argument("--skip_wan", action="store_true",
                    help="Skip the large Wan2.1-I2V model download")
parser.add_argument("--token", default=os.environ.get("HF_TOKEN"),
                    help="HuggingFace token (falls back to HF_TOKEN env var)")
args = parser.parse_args()

if args.token:
    login(token=args.token)

os.makedirs("./checkpoints", exist_ok=True)

# Small checkpoints from HF
repo_id_list   = ["Ruicheng/moge-vitl", "Iceclear/StableSR", "Iceclear/StableSR",
                  "Skywork/Matrix-3D", "Skywork/Matrix-3D", "Skywork/Matrix-3D",
                  "Skywork/Matrix-3D", "Skywork/Matrix-3D"]
filename_list  = ["model.pt", "stablesr_turbo.ckpt", "vqgan_cfw_00011.ckpt",
                  "checkpoints/text2panoimage_lora.safetensors", "checkpoints/pano_lrm_480p.pt",
                  "checkpoints/pano_video_gen_480p.ckpt", "checkpoints/pano_video_gen_720p.bin",
                  "checkpoints/pano_video_gen_720p_5b.safetensors"]
local_dir_list = ["./checkpoints/moge", "./checkpoints/StableSR", "./checkpoints/StableSR",
                  "./checkpoints/flux_lora", "./checkpoints/pano_lrm",
                  "./checkpoints/Wan-AI/wan_lora", "./checkpoints/Wan-AI/wan_lora",
                  "./checkpoints/Wan-AI/wan_lora"]

for repo_id, filename, local_dir in zip(repo_id_list, filename_list, local_dir_list):
    print(f"\nDownloading {filename} from {repo_id} → {local_dir}...\n")
    download_ckpt(local_dir, repo_id, filename)

# VideoLLaMA3-7B (prompt generation for i2p mode) — to HF cache
print("\nDownloading DAMO-NLP-SG/VideoLLaMA3-7B to HF cache...\n")
snapshot_download("DAMO-NLP-SG/VideoLLaMA3-7B")

# FLUX.1-Fill-dev (gated — requires HF token + accepted terms)
# Downloaded to HF cache (not local_dir) so from_pretrained() finds it automatically
print("\nDownloading black-forest-labs/FLUX.1-Fill-dev to HF cache...\n")
snapshot_download("black-forest-labs/FLUX.1-Fill-dev")

# Large Wan2.1-I2V base model
if not args.skip_wan:
    if args.resolution in ("720p", "both"):
        download_snapshot("./checkpoints/Wan-AI/Wan2.1-I2V-14B-720P", "Wan-AI/Wan2.1-I2V-14B-720P")
    if args.resolution in ("480p", "both"):
        download_snapshot("./checkpoints/Wan-AI/Wan2.1-I2V-14B-480P", "Wan-AI/Wan2.1-I2V-14B-480P")
