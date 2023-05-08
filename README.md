### Windows

- 建议下载`miniconda`或者`python 3.10`
- 点击下载：[miniconda](https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe)或[python 3.10](https://www.python.org/ftp/python/3.10.10/python-3.10.10-amd64.exe)
- 下载模型到`models`文件夹中后运行`环境下载.bat`，等待所需环境安装完成后可自行输入网址，下次运行点击`运行.bat`即可使用
- 模型下载地址：https://huggingface.co/BlinkDL/rwkv-4-novel/tree/main


### Linux

- 下载所需环境后，`wget`模型到`models`文件夹后，`python demo.py`即可



### 注意事项

- torch版本不匹配或者下到cpu版本了Windows请运行`torch有问题请运行这个.bat`，Linux请下载`torch torchvision  --index-url https://download.pytorch.org/whl/cu118`