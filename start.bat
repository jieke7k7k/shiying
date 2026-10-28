@echo off
REM Shiying-3.2.0 · Windows 双击启动入口
REM 双击本文件打开 app_native.py 的原生窗口（pywebview，缺省回退
REM Chrome/Edge 的 --app 应用窗口）；用完后点页面“停止服务”，
REM 窗口与服务会一起关闭，不驻留后台。
REM
REM 3.2.0：
REM   - chcp 65001 + PYTHONUTF8=1：中文文件夹与中文照片名不乱码；
REM   - 启动前检查 Pillow，缺依赖时自动 pip install -r requirements.txt，
REM     装完仍缺才报错停住；
REM   - 任何失败路径末尾都有 pause，窗口绝不一闪而过；
REM   - 单实例锁：已有实例在运行时 app_native 会提示并退出，不会开第二个窗口。
REM 注意：本机 Linux 无法执行 .bat，本文件只做内容审查，
REM 须在 Windows 真机上实测。
chcp 65001 >nul
set PYTHONUTF8=1
title Shiying-3.2.0
cd /d "%~dp0"

REM ---- 找 Python（优先 py 启动器，其次 python）----
set "PY="
where py >nul 2>nul
if not errorlevel 1 set "PY=py -3"
if "%PY%"=="" (
  where python >nul 2>nul
  if not errorlevel 1 set "PY=python"
)
if "%PY%"=="" (
  echo.
  echo 未找到 Python（需要 Python 3.9+，请先安装并勾选“Add Python to PATH”）。
  echo 按任意键关闭本窗口。
  pause >nul
  exit /b 1
)

REM ---- 缺依赖时自动安装（只需成功一次）----
%PY% -c "import PIL" >nul 2>nul
if errorlevel 1 (
  echo 正在安装依赖（只需一次）：%PY% -m pip install -r requirements.txt
  %PY% -m pip install -r requirements.txt
  %PY% -c "import PIL" >nul 2>nul
  if errorlevel 1 (
    echo.
    echo 依赖安装失败：Pillow 仍不可用。请检查网络后手动运行：
    echo   %PY% -m pip install -r requirements.txt
    echo 按任意键关闭本窗口。
    pause >nul
    exit /b 1
  )
)

REM ---- 启动原生窗口（失败时停住，不闪退）----
REM 退出码区分：2 = 单实例锁拦截（已有窗口在运行），1 = 启动失败。
REM 注意批处理里 if errorlevel 2 要写在 if errorlevel 1 之前，
REM 因为 errorlevel 1 对所有 >=1 的退出码都成立。
%PY% app_native.py
if errorlevel 2 (
  echo.
  echo 已有拾影窗口在运行，请先看已打开的窗口，不要重复打开。
  echo 按任意键关闭本窗口。
  pause >nul
  exit /b 2
)
if errorlevel 1 (
  echo.
  echo 启动失败，见上方提示。若提示没有原生窗口，可先运行：
  echo   %PY% -m pip install pywebview
  echo 按任意键关闭本窗口。
  pause >nul
)
