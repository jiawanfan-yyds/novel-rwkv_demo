python -m venv novel-rmkv
call novel-rmkv\Scripts\activate.bat
pip install -r requirements.txt
start "" cmd /k "python demo.py"
start http://127.0.0.1:7777
