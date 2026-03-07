@echo off
setlocal
set "CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
set "PATH=%CUDA_HOME%\bin;%CUDA_HOME%\libnvvp;%PATH%"
set "PYTHONNOUSERSITE=1"
set "PIP_USE_PEP517=0"

echo ✅ Installing Submodules...

pushd submodules\nvdiffrast || goto :error
pip install . || goto :error
popd

pushd submodules\simple-knn || goto :error
python setup.py install
if errorlevel 1 (
    echo [WARN] simple-knn native build failed on this Windows environment. Continuing without it.
)
popd

pip install git+https://github.com/rmurai0610/diff-gaussian-rasterization-w-pose.git
if errorlevel 1 (
    echo [WARN] diff-gaussian-rasterization-w-pose build failed on this environment. Continuing without it.
)

if not exist submodules\ODGS (
    git clone https://github.com/esw0116/ODGS.git submodules\ODGS || goto :error
)

pushd submodules\ODGS || goto :error
pip install submodules/odgs-gaussian-rasterization
if errorlevel 1 (
    echo [WARN] odgs-gaussian-rasterization build failed. Continuing without ODGS native rasterization.
)
popd

pushd code\DiffSynth-Studio || goto :error
echo ✅ Installing DiffSynth-Studio...
python setup.py develop
if errorlevel 1 (
    echo [WARN] setup.py develop failed, trying setup.py install...
    python setup.py install || goto :error
)
popd

echo ✅ Installing Python dependencies...
pip install "numpy<2" "opencv-python<4.11" || goto :error
pip install plyfile decord ffmpeg trimesh pyrender xfuser diffusers open3d py360convert || goto :error
pip install "git+https://github.com/facebookresearch/pytorch3d.git@v0.7.7" || goto :error
pip install peft easydict torchsde "open-clip-torch==2.7.0" fairscale natsort || goto :error
pip install realesrgan || goto :error
pip install "flash-attn==2.7.4.post1" --no-build-isolation || goto :error
pip install git+https://github.com/EasternJournalist/utils3d.git#egg=utils3d || goto :error
pip install "xformers==0.0.31" || goto :error
pip install "jaxtyping==0.3.2" || goto :error
pip install "modelscope==1.28.2" || goto :error
pip install "diffusers==0.35.1" || goto :error
pip install "matplotlib==3.8.4" || goto :error
pip install "transformers==4.56.0" || goto :error
pip install "torchmetrics==0.7.0" || goto :error
pip install "OmegaConf==2.1.1" || goto :error
pip install "imageio-ffmpeg==0.6.0" || goto :error
pip install "pytorch-lightning==1.4.2" || goto :error
pip install "omegaconf==2.1.1" || goto :error
pip install "webdataset==0.2.5" || goto :error
pip install "kornia==0.6" || goto :error
pip install "streamlit==1.12.1" || goto :error
pip install "einops==0.8.0" || goto :error
pip install open_clip_torch || goto :error
pip install "SwissArmyTransformer==0.4.12" || goto :error
pip install "wandb==0.21.1" || goto :error
pip install -e git+https://github.com/CompVis/taming-transformers.git@master#egg=taming-transformers || goto :error
pip uninstall -y basicsr || goto :error
pip install openai-clip || goto :error

echo ✅ All dependencies installed successfully.
goto :eof

:error
echo ❌ Installation failed. Aborting.
exit /b 1
