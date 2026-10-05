@echo off
cd /d %USERPROFILE%\Documents\kimi\workspace\cc-eval
%USERPROFILE%\Documents\kimi\workspace\cc-eval\.venv\Scripts\python.exe injection\eval_base_seen.py > injection\base_seen_run.log 2>&1
