echo "✅ Installing Submodules..."
cd ./submodules/nvdiffrast/
pip install .

cd ../simple-knn/
python setup.py install

pip install git+https://github.com/rmurai0610/diff-gaussian-rasterization-w-pose.git
git clone https://github.com/esw0116/ODGS.git
cd ../ODGS
pip install submodules/odgs-gaussian-rasterization
cd ../..

cd code
echo "✅ Installing DiffSynth-Studio..."
cd DiffSynth-Studio/
pip install -e .
cd ..

echo "✅ Installing Python dependencies..."
pip install plyfile decord ffmpeg trimesh pyrender xfuser diffusers open3d py360convert
pip install "git+https://github.com/facebookresearch/pytorch3d.git@v0.7.7"
pip install peft easydict torchsde open-clip-torch>=2.7.0 fairscale natsort
pip install realesrgan #Version >3.7 and <3.9
pip install flash-attn>=2.7.4.post1 --no-build-isolation
pip install git+https://github.com/EasternJournalist/utils3d.git#egg=utils3d
pip install xformers>=0.0.31
pip install jaxtyping>=0.3.2
pip install modelscope>=1.28.2
pip install diffusers>=0.34.0
pip install matplotlib>=3.8.4
pip install transformers>=4.56.0
pip install torchmetrics>=0.7.0
pip install OmegaConf>=2.1.1
pip install imageio-ffmpeg>=0.6.0
pip install pytorch-lightning>=1.4.2
pip install omegaconf>=2.1.1
pip install webdataset>=0.2.5
pip install kornia>=0.6
pip install streamlit>=1.12.1
pip install einops>=0.8.0
pip install open_clip_torch
pip install SwissArmyTransformer>=0.4.12
pip install wandb>=0.21.1
pip install -e git+https://github.com/CompVis/taming-transformers.git@master#egg=taming-transformers
pip uninstall basicsr
pip install openai-clip


echo "✅ All dependencies installed successfully."
