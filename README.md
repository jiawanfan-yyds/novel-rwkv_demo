### Windows

- 建议下载`miniconda`或者`python 3.10`
- 点击下载：[miniconda](https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe)或[python 3.10](https://www.python.org/ftp/python/3.10.10/python-3.10.10-amd64.exe)
- 下载模型到`models`文件夹中后运行`环境下载.bat`，下次运行点击`运行.bat`即可使用
- 模型下载地址：https://huggingface.co/BlinkDL/rwkv-4-novel/tree/main


### Linux

- 下载所需环境后，`wget`模型到`models`文件夹后，`python demo.py`即可



### 注意事项

- 请将`model_default.json`文件中的模型名字更改为自己所下的模型名，`strategy_default`默认为`cuda fp16i8`（需要9G多的显存），请根据自己的实际配置修改

- torch版本不匹配或者下到cpu版本了Windows请运行`torch有问题请运行这个.bat`，Linux请下载`torch torchvision  --index-url https://download.pytorch.org/whl/cu118`