"""nnUNetv2 数据处理脚本：Vesuvius Challenge Surface Detection。

本脚本从 `train_nnunet.py` 中抽离“数据集准备/预处理”部分，用于把流程拆成更清晰的两步：

Step 1: 准备 nnUNet_raw 数据集
  - 将比赛的 train_images/train_labels 转换为 nnUNet 所需结构（TIFF + JSON spacing sidecar）

Step 2: nnUNetv2_plan_and_preprocess
  - 运行 nnUNetv2 的规划与预处理，生成 nnUNet_preprocessed

注意：
- 本脚本不会导入或调用 `train_nnunet.py`；两者是独立脚本。
- Windows 下若 symlink 权限受限，可加 --no-symlinks，自动回退为 copy。
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from multiprocessing import get_context
from pathlib import Path
from typing import Optional, Sequence


def is_windows() -> bool:
	return os.name == "nt"


def run(cmd: str, *, timeout: Optional[int] = None) -> None:
	print(f"Running: {cmd}")
	if timeout is not None:
		print(f"Timeout: {timeout}s ({timeout/3600:.1f}h)")
	completed = subprocess.run(cmd, shell=True, timeout=timeout)
	if completed.returncode != 0:
		raise RuntimeError(f"Command failed (exit={completed.returncode}): {cmd}")


def safe_symlink_or_copy(src: Path, dst: Path) -> None:
	"""尽量 symlink；失败则 copy。"""
	dst.parent.mkdir(parents=True, exist_ok=True)
	if dst.exists():
		return

	try:
		dst.symlink_to(src.resolve())
	except Exception:
		shutil.copy2(src, dst)


def write_spacing_json(json_path: Path, spacing: Sequence[float] = (1.0, 1.0, 1.0)) -> None:
	json_path.parent.mkdir(parents=True, exist_ok=True)
	json_path.write_text(json.dumps({"spacing": list(spacing)}), encoding="utf-8")


def create_dataset_json(dataset_dir: Path, num_training: int, *, file_ending: str = ".tif") -> None:
	"""创建 nnUNet dataset.json（启用 ignore label，并指定 SimpleTiffIO）。"""
	dataset_json = {
		"channel_names": {"0": "CT"},
		"labels": {"background": 0, "surface": 1, "ignore": 2},
		"numTraining": int(num_training),
		"file_ending": file_ending,
		"overwrite_image_reader_writer": "SimpleTiffIO",
	}
	(dataset_dir / "dataset.json").write_text(json.dumps(dataset_json, indent=4), encoding="utf-8")


def _iter_tifs(folder: Path) -> list[Path]:
	return sorted(folder.glob("*.tif"))


def _prepare_case(
	img_path: Path,
	train_labels_dir: Path,
	images_tr_dir: Path,
	labels_tr_dir: Path,
	use_symlinks: bool,
	spacing: Sequence[float],
) -> bool:
	case_id = img_path.stem
	label_path = train_labels_dir / img_path.name
	if not label_path.exists():
		return False

	# nnUNet: imagesTr/case_xxx_0000.tif, labelsTr/case_xxx.tif
	dst_img = images_tr_dir / f"{case_id}_0000.tif"
	dst_img_json = images_tr_dir / f"{case_id}_0000.json"
	dst_lbl = labels_tr_dir / f"{case_id}.tif"
	dst_lbl_json = labels_tr_dir / f"{case_id}.json"

	try:
		if use_symlinks:
			safe_symlink_or_copy(img_path, dst_img)
			safe_symlink_or_copy(label_path, dst_lbl)
		else:
			dst_img.parent.mkdir(parents=True, exist_ok=True)
			dst_lbl.parent.mkdir(parents=True, exist_ok=True)
			if not dst_img.exists():
				shutil.copy2(img_path, dst_img)
			if not dst_lbl.exists():
				shutil.copy2(label_path, dst_lbl)

		if not dst_img_json.exists():
			write_spacing_json(dst_img_json, spacing)
		if not dst_lbl_json.exists():
			write_spacing_json(dst_lbl_json, spacing)
		return True
	except Exception as exc:
		print(f"[WARN] prepare failed: {img_path.name}: {exc}")
		return False


def prepare_dataset(
	*,
	input_dir: Path,
	nnunet_raw_dir: Path,
	dataset_name: str,
	max_cases: Optional[int],
	num_workers: int,
	use_symlinks: bool,
	spacing: Sequence[float],
) -> Path:
	"""将比赛数据转换为 nnUNet raw 数据集结构（TIFF 直读）。"""
	train_images_dir = input_dir / "train_images"
	train_labels_dir = input_dir / "train_labels"
	if not train_images_dir.exists():
		raise FileNotFoundError(f"train_images not found: {train_images_dir}")
	if not train_labels_dir.exists():
		raise FileNotFoundError(f"train_labels not found: {train_labels_dir}")

	dataset_dir = nnunet_raw_dir / dataset_name
	images_tr_dir = dataset_dir / "imagesTr"
	labels_tr_dir = dataset_dir / "labelsTr"
	images_tr_dir.mkdir(parents=True, exist_ok=True)
	labels_tr_dir.mkdir(parents=True, exist_ok=True)

	image_files = _iter_tifs(train_images_dir)
	if max_cases is not None:
		image_files = image_files[:max_cases]
	print(f"Found {len(image_files)} training cases")
	print(f"Data mode: {'symlinks (fallback copy)' if use_symlinks else 'copy'}")
	print(f"Workers: {num_workers}")

	if num_workers <= 1:
		results = [
			_prepare_case(
				p,
				train_labels_dir,
				images_tr_dir,
				labels_tr_dir,
				use_symlinks=use_symlinks,
				spacing=spacing,
			)
			for p in image_files
		]
	else:
		# Windows 下强制 spawn；Linux 默认 fork（更快），但在不支持 fork 的平台回退到 spawn
		if is_windows():
			ctx = get_context("spawn")
		else:
			try:
				ctx = get_context("fork")
			except ValueError:
				ctx = get_context("spawn")
		with ctx.Pool(processes=num_workers) as pool:
			results = pool.starmap(
				_prepare_case,
				[
					(
						p,
						train_labels_dir,
						images_tr_dir,
						labels_tr_dir,
						use_symlinks,
						spacing,
					)
					for p in image_files
				],
				chunksize=8,
			)

	num_converted = int(sum(bool(x) for x in results))
	create_dataset_json(dataset_dir, num_converted)
	print(f"Dataset prepared: {num_converted} cases")
	print(f"Dataset dir: {dataset_dir}")
	return dataset_dir


def link_prepared_preprocessed(
	*,
	prepared_path: Path,
	nnunet_preprocessed_dir: Path,
	dataset_name: str,
) -> bool:
	"""链接/复制已缓存的 nnUNet_preprocessed（可跳过 plan_and_preprocess）。"""
	if not prepared_path.exists():
		return False

	# 兼容：prepared_path 直接是 DatasetXXX_* 或其父目录
	source_dir = prepared_path
	if not (source_dir / "dataset.json").exists():
		candidates = list(prepared_path.glob("Dataset*"))
		if not candidates:
			return False
		source_dir = candidates[0]

	target_dir = nnunet_preprocessed_dir / dataset_name
	if target_dir.exists():
		print(f"Preprocessed already exists: {target_dir}")
		return True

	print(f"Linking prepared preprocessed from: {source_dir}")
	target_dir.mkdir(parents=True, exist_ok=True)

	copy_suffixes = {".json", ".pkl", ".txt"}
	link_suffixes = {".npz", ".npy", ".b2nd"}
	copied = 0
	linked = 0

	for src in source_dir.rglob("*"):
		if src.is_dir():
			continue
		rel = src.relative_to(source_dir)
		dst = target_dir / rel
		dst.parent.mkdir(parents=True, exist_ok=True)

		if src.suffix.lower() in link_suffixes:
			safe_symlink_or_copy(src, dst)
			linked += 1
		elif src.suffix.lower() in copy_suffixes:
			if not dst.exists():
				shutil.copy2(src, dst)
				copied += 1
		else:
			# 其它小文件：复制更稳
			if not dst.exists():
				shutil.copy2(src, dst)
				copied += 1

	print(f"Prepared preprocessed: copied={copied}, linked_or_copied={linked}")
	return True


def set_nnunet_env(*, nnunet_raw: Path, nnunet_preprocessed: Path, nnunet_results: Path, compile_flag: bool) -> None:
	nnunet_raw.mkdir(parents=True, exist_ok=True)
	nnunet_preprocessed.mkdir(parents=True, exist_ok=True)
	nnunet_results.mkdir(parents=True, exist_ok=True)
	os.environ["nnUNet_raw"] = str(nnunet_raw)
	os.environ["nnUNet_preprocessed"] = str(nnunet_preprocessed)
	os.environ["nnUNet_results"] = str(nnunet_results)
	os.environ["nnUNet_compile"] = "true" if compile_flag else "false"


def nnunet_preprocess(
	*,
	dataset_id: int,
	planner: str,
	configuration: str,
	num_workers: int,
	timeout: Optional[int],
) -> None:
	cmd = (
		f"nnUNetv2_plan_and_preprocess -d {dataset_id:03d} -np {num_workers} "
		f"-pl {planner} -c {configuration}"
	)
	run(cmd, timeout=timeout)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Vesuvius nnUNetv2 dataset prepare + plan_and_preprocess")
	parser.add_argument("--input-dir", type=Path, required=True, help="比赛数据根目录（包含 train_images/train_labels）")
	parser.add_argument("--work-dir", type=Path, default=Path("./work"), help="工作目录（默认 ./work）")
	parser.add_argument("--nnunet-raw", type=Path, default=None, help="覆盖 nnUNet_raw 路径")
	parser.add_argument("--nnunet-preprocessed", type=Path, default=None, help="覆盖 nnUNet_preprocessed 路径")
	parser.add_argument("--nnunet-results", type=Path, default=None, help="覆盖 nnUNet_results 路径")

	parser.add_argument("--dataset-id", type=int, default=100)
	parser.add_argument(
		"--dataset-name",
		type=str,
		default=None,
		help="nnUNet 数据集目录名（默认: Dataset{ID:03d}_VesuviusSurface）",
	)

	parser.add_argument("--configuration", type=str, default="3d_fullres")
	parser.add_argument("--planner", type=str, default="nnUNetPlannerResEncM")
	parser.add_argument("--num-workers", type=int, default=max(os.cpu_count() or 4, 1))
	parser.add_argument("--max-cases", type=int, default=None)
	parser.add_argument("--no-symlinks", action="store_true", help="禁用 symlink（强制 copy）")
	parser.add_argument("--spacing", type=float, nargs=3, default=(1.0, 1.0, 1.0))
	parser.add_argument("--timeout", type=int, default=None)
	parser.add_argument("--compile", type=int, default=0, choices=[0, 1], help="是否设置 nnUNet_compile=true")

	# 两步拆分开跑
	parser.add_argument("--do-prepare", type=int, default=1, choices=[0, 1])
	parser.add_argument("--do-preprocess", type=int, default=1, choices=[0, 1])

	parser.add_argument(
		"--prepared-preprocessed",
		type=Path,
		default=None,
		help="已缓存的 preprocessed 路径（可配合 --do-preprocess 0）",
	)
	return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
	args = parse_args(argv)

	work_dir: Path = args.work_dir
	nnunet_raw = args.nnunet_raw or (work_dir / "nnUNet_raw")
	nnunet_preprocessed = args.nnunet_preprocessed or (work_dir / "nnUNet_preprocessed")
	nnunet_results = args.nnunet_results or (work_dir / "nnUNet_results")

	dataset_name = args.dataset_name or f"Dataset{args.dataset_id:03d}_VesuviusSurface"

	set_nnunet_env(
		nnunet_raw=nnunet_raw,
		nnunet_preprocessed=nnunet_preprocessed,
		nnunet_results=nnunet_results,
		compile_flag=bool(args.compile),
	)

	print("=" * 60)
	print("nnUNet data processing")
	print("=" * 60)
	print("input_dir:", args.input_dir)
	print("dataset_id:", args.dataset_id)
	print("dataset_name:", dataset_name)
	print("configuration:", args.configuration)
	print("planner:", args.planner)
	print("num_workers:", args.num_workers)
	print("nnUNet_raw:", nnunet_raw)
	print("nnUNet_preprocessed:", nnunet_preprocessed)
	print("nnUNet_results:", nnunet_results)
	print("=" * 60)

	if args.do_prepare:
		prepare_dataset(
			input_dir=args.input_dir,
			nnunet_raw_dir=nnunet_raw,
			dataset_name=dataset_name,
			max_cases=args.max_cases,
			num_workers=max(int(args.num_workers), 1),
			use_symlinks=not args.no_symlinks,
			spacing=args.spacing,
		)

	# 如果用户提供了 prepared preprocessed，则先尝试链接（可配合 --do-preprocess 0）
	if args.prepared_preprocessed is not None:
		linked = link_prepared_preprocessed(
			prepared_path=args.prepared_preprocessed,
			nnunet_preprocessed_dir=nnunet_preprocessed,
			dataset_name=dataset_name,
		)
		print(f"prepared preprocessed linked: {linked}")

	if args.do_preprocess:
		nnunet_preprocess(
			dataset_id=args.dataset_id,
			planner=args.planner,
			configuration=args.configuration,
			num_workers=max(int(args.num_workers), 1),
			timeout=args.timeout,
		)

	print("Done.")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
