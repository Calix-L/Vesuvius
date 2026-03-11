#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Kaggle 数据集/比赛数据下载：断网自动等待 + 自动重试 + 断点续传
- 支持 dataset:  kaggle datasets download -d <owner/dataset>
- 支持 competition: kaggle competitions download -c <competition>

特点：
- 断网/超时/429/5xx 等自动重试
- 每次失败后等待网络恢复再继续
- 使用 --resume 进行续传（zip 分卷/未完成时继续）
- 可选自动解压
"""

from __future__ import annotations

import argparse
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional


def log(msg: str) -> None:
    t = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print(f"[{t}] {msg}", flush=True)


def is_network_ok(host: str = "www.google.com", port: int = 443, timeout: float = 3.0) -> bool:
    """
    用 TCP 连接判断网络是否通（比 ping 更通用；很多环境禁 ping）。
    你也可以改成试连 kaggle.com:443。
    """
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def wait_for_network(check_host: str, check_port: int, interval: int) -> None:
    while True:
        if is_network_ok(check_host, check_port):
            return
        log(f"网络不可用，{interval}s 后重试... (check={check_host}:{check_port})")
        time.sleep(interval)


def run_cmd(cmd: list[str], cwd: Optional[Path] = None) -> tuple[int, str]:
    """
    运行命令并返回 (returncode, combined_output)
    """
    p = subprocess.Popen(
        cmd,
        cwd=str(cwd) if cwd else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    out, _ = p.communicate()
    return p.returncode, out


def should_retry(output: str, code: int) -> bool:
    """
    判断是否属于“值得重试”的错误。
    Kaggle CLI 在断网/429/5xx/SSL 等情况下通常会输出特征字符串。
    """
    o = output.lower()
    retry_signals = [
        "connection", "timed out", "timeout", "temporarily unavailable",
        "ssl", "tls", "reset by peer", "remote end closed",
        "429", "too many requests",
        "500", "502", "503", "504",
        "max retries", "rate limit",
        "name or service not known", "nodename nor servname",
        "failed to establish a new connection",
        "connection aborted", "connection refused",
        "chunked encoding error",
    ]
    if code == 0:
        return False
    return any(s in o for s in retry_signals)


def main() -> None:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--dataset", help="Kaggle dataset slug: owner/dataset-name")
    g.add_argument("--competition", help="Kaggle competition name: xxx-yyy")

    ap.add_argument("--out", default=".", help="下载输出目录（默认当前目录）")
    ap.add_argument("--unzip", action="store_true", help="下载完成后自动解压")
    ap.add_argument("--max-retries", type=int, default=0,
                    help="最大重试次数；0 表示无限重试（默认）")
    ap.add_argument("--base-sleep", type=int, default=10,
                    help="基础等待秒数（失败后指数退避起点）")
    ap.add_argument("--max-sleep", type=int, default=300,
                    help="最大等待秒数（指数退避上限）")
    ap.add_argument("--net-check-host", default="kaggle.com", help="网络检测主机（默认 kaggle.com）")
    ap.add_argument("--net-check-port", type=int, default=443, help="网络检测端口（默认 443）")
    ap.add_argument("--net-check-interval", type=int, default=10, help="断网时轮询间隔秒（默认 10）")

    args = ap.parse_args()
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # 构造 kaggle 命令
    if args.dataset:
        cmd = ["kaggle", "datasets", "download", "-d", args.dataset, "-p", str(out_dir), "--resume"]
    else:
        cmd = ["kaggle", "competitions", "download", "-c", args.competition, "-p", str(out_dir), "--resume"]

    if args.unzip:
        cmd.append("--unzip")

    log(f"输出目录: {out_dir}")
    log("准备开始下载（支持断点续传 --resume）")
    log("命令: " + " ".join(cmd))

    # 先等网络可用
    wait_for_network(args.net_check_host, args.net_check_port, args.net_check_interval)

    attempt = 0
    sleep_s = args.base_sleep

    while True:
        attempt += 1
        log(f"开始尝试 #{attempt}")
        code, out = run_cmd(cmd)
        # 打印输出（防止 Kaggle CLI 把关键信息只写 stdout）
        if out.strip():
            print(out, end="" if out.endswith("\n") else "\n", flush=True)

        if code == 0:
            log("✅ 下载完成")
            return

        # 不可重试就直接退出（例如权限/slug 错等）
        if not should_retry(out, code):
            log("❌ 失败且看起来不是网络/临时错误（不自动重试）。")
            log(f"退出码: {code}")
            sys.exit(code)

        # 可重试：检查网络 -> 等待 -> 重试（指数退避）
        if args.max_retries and attempt >= args.max_retries:
            log(f"❌ 达到最大重试次数 max_retries={args.max_retries}，退出。")
            sys.exit(code)

        log("检测到可重试错误，先确认网络是否恢复...")
        wait_for_network(args.net_check_host, args.net_check_port, args.net_check_interval)

        log(f"等待 {sleep_s}s 后重试（指数退避，上限 {args.max_sleep}s）...")
        time.sleep(sleep_s)
        sleep_s = min(args.max_sleep, max(args.base_sleep, sleep_s * 2))


if __name__ == "__main__":
    main()
