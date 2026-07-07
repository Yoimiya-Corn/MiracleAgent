@echo off
REM CoreCoder Web Server 启动脚本
REM 先复制 .env.example 为 .env 并修改 vLLM 地址

cd /d "%~dp0"

REM 检查依赖
pip install -e ".[web]" -q 2>nul

REM 启动
python web_server.py --host 0.0.0.0 --port 8080 %*
