@echo off
setlocal
set "CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
set "PATH=%CUDA_HOME%\bin;%CUDA_HOME%\libnvvp;%PATH%"
set "USE_LIBUV=0"
set "HF_TOKEN=c3876e09447f08f0aebb615ede24f61030b14b32"

python code/vbench_batch.py ^
    --output_dir=output\vbench\videos ^
    --num_samples=5 ^
    --seed=0 ^
    --resolution=720

echo Generation completed.
endlocal
