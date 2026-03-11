#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import sys
import zipfile
from pathlib import Path

def ensure_kaggle_auth():
    """
    检查 kaggle 认证信息是否存在：~/.kaggle/kaggle.json 或环境变量
    """
    has_env = bool(os.getenv("KAGGLE_USERNAME")) and bool(os.getenv("KAGGLE_KEY"))
    has_json = (Path.home() / ".kaggle" / "kaggle.json").exists()
    if not (has_env or has_json):
        raise RuntimeError(
            "未检测到 Kaggle 认证信息。\n"
            "请在服务器上放置 ~/.kaggle/kaggle.json 并 chmod 600，\n"
            "或设置环境变量 KAGGLE_USERNAME / KAGGLE_KEY。"
        )

def unzip_file(zip_path: Path, out_dir: Path):
    print(f"[INFO] 解压: {zip_path} -> {out_dir}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(out_dir)

def main():
    parser = argparse.ArgumentParser(
        description="Download Kaggle competition data and unzip it."
    )
    parser.add_argument(
        "--comp",
        default="vesuvius-challenge-surface-detection",
        help="competition slug, default: vesuvius-challenge-surface-detection",
    )
    parser.add_argument(
        "--out",
        default="./kaggle_data/vesuvius_surface_detection",
        help="output directory",
    )
    parser.add_argument(
        "--unzip",
        action="store_true",
        help="unzip downloaded zip(s)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="redownload even if zip exists",
    )
    args = parser.parse_args()

    ensure_kaggle_auth()

    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        print("[ERROR] 未安装 kaggle 包。请先运行：pip install -U kaggle", file=sys.stderr)
        sys.exit(1)

    api = KaggleApi()
    api.authenticate()

    # Kaggle 官方接口：下载比赛数据（通常会得到一个 zip）
    print(f"[INFO] 开始下载比赛数据: {args.comp}")
    print(f"[INFO] 输出目录: {out_dir}")

    # kaggle API 会根据 comp 下载 zip 文件到 path
    # force 参数可以强制覆盖下载
    api.competition_download_files(
        competition=args.comp,
        path=str(out_dir),
        force=args.force,
        quiet=False,
    )

    # 找到下载的 zip
    zips = sorted(out_dir.glob("*.zip"))
    if not zips:
        print("[WARN] 没找到 zip 文件。可能下载失败或 Kaggle 返回了不同的文件名。", file=sys.stderr)
        sys.exit(2)

    print("[INFO] 下载完成：")
    for z in zips:
        print("  -", z.name)

    if args.unzip:
        for z in zips:
            unzip_file(z, out_dir)
        print("[INFO] 解压完成。")

if __name__ == "__main__":
    main()
