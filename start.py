#!/usr/bin/env python3
"""Shiying-3.2.0 的浏览器回退启动脚本（legacy）。

3.2.0 的正式入口是 app_native.py（原生窗口、单实例、干净关闭协议）。
本文件保留给“不想装 pywebview、也不需要无边框窗口”的场景：双击
（配合 start.bat）或 python3 start.py 会起服务并用默认浏览器打开。

流程：
  1) 检查 Pillow 是否可用（缺失则打印安装命令并退出，不静默乱装）；
  2) 以子进程启动 server.py（本地服务，只监听 127.0.0.1）；
  3) 读到服务地址后自动打开浏览器；
  4) 前台等待：页面点“停止服务”会让 server.py 自己退出，
     本脚本随之退出；本窗口按 Ctrl+C 或直接关窗口也会停掉服务，
     不驻留后台。

用法：
  python3 start.py [--port 8765] [--no-browser]
"""
import subprocess
import sys
import time
import os

TOOL_DIR = os.path.dirname(os.path.abspath(__file__))
SERVER_PY = os.path.join(TOOL_DIR, "server.py")


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=0)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args(argv)

    try:
        import PIL  # noqa: F401
    except ImportError:
        print("缺少 Pillow（图片处理库），请先安装依赖再启动：")
        print("  pip install -r requirements.txt")
        print("（requirements.txt 就在本文件夹里；Windows 上用 py -m pip install -r requirements.txt）")
        return 1
    if not os.path.exists(SERVER_PY):
        print("找不到 server.py：%s" % SERVER_PY)
        return 1

    cmd = [sys.executable, SERVER_PY]
    if args.port:
        cmd += ["--port", str(args.port)]
    cmd += ["--no-open"]  # 浏览器由本脚本统一打开，避免重复打开
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True,
                            cwd=TOOL_DIR)
    url = None
    try:
        # 读启动日志，直到拿到地址（最多等约 15 秒；用单调时钟，系统改时间不影响）
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            line = proc.stdout.readline()
            if not line:
                break
            sys.stdout.write(line)
            sys.stdout.flush()
            if "已有实例在运行" in line:
                print("检测到已有实例在运行，本次启动不再开新端口。")
                try:
                    proc.wait(timeout=5)
                except Exception:
                    pass
                return 1
            if "http://127.0.0.1:" in line:
                url = line.strip().split("：")[-1].split(": ")[-1]
                # 更稳妥：直接从行里截取 http 开头的部分
                start = line.find("http://127.0.0.1:")
                if start >= 0:
                    url = line[start:].strip()
                break
        if proc.poll() is not None:
            print("服务启动失败（已退出，退出码 %s），见上方日志。" % proc.returncode)
            return 1
        if url and not args.no_browser:
            try:
                import webbrowser
                webbrowser.open(url)
            except Exception as e:
                print("自动打开浏览器失败，请手动打开：%s（%s）" % (url, e))
        elif url:
            print("服务地址：%s（已按 --no-browser 跳过自动打开）" % url)
        else:
            print("未读到服务地址，但服务进程仍在运行；请看上方日志里的地址。")
        print("关闭方式：页面点“停止服务”，或本窗口按 Ctrl+C，或直接关掉本窗口。")
        # 把服务剩余日志透传到本窗口，直到服务退出
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
        proc.wait()
    except KeyboardInterrupt:
        print("\n正在停止服务……")
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
    print("服务已停止，无后台残留。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
