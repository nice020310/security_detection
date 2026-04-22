@echo off
setlocal
call "D:\anaconda\Scripts\activate.bat" bqs1
cd /d "E:\毕业论文\code\security_detection"
echo Using Python: %CONDA_PREFIX%\python.exe
python -V
python -u 1.py