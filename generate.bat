@echo off
setlocal
set "CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
set "PATH=%CUDA_HOME%\bin;%CUDA_HOME%\libnvvp;%PATH%"
set "USE_LIBUV=0"

set "output_dir=output\example1"

REM Step1: text to panorama image
python code/panoramic_image_generation.py ^
    --mode=t2p ^
    --prompt="a medieval village, half-timbered houses, cobblestone streets, lush greenery, clear blue sky, detailed textures, vibrant colors, high resolution" ^
    --output_path="%output_dir%"

REM Or you can choose image to panorama image generation
REM python code/panoramic_image_generation.py ^
REM     --mode=i2p ^
REM     --input_image_path="./data/image2.jpg" ^
REM     --output_path="%output_dir%"

REM Step2: panorama image to video generation
set "VISIBLE_GPU_NUM=1"
torchrun --nproc_per_node %VISIBLE_GPU_NUM% code/panoramic_image_to_video.py ^
    --inout_dir="%output_dir%" ^
    --resolution=720

REM Step3: 3d scene extraction
python code/panoramic_video_to_3DScene.py ^
    --inout_dir="%output_dir%" ^
    --resolution=720

echo ✅ Generation pipeline commands completed.
endlocal
