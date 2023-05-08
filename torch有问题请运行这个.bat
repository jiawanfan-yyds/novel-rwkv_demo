call novel-rmkv\Scripts\activate.bat
@echo off
echo Downloading PyTorch...
torch torchvision  --index-url https://download.pytorch.org/whl/cu118
chcp 65001
echo Download completed.
msg * 下载完成
