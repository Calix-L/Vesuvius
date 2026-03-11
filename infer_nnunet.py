"""nnUNetv2 推理 + 生成 submission.zip（Vesuvius Challenge）

职责：
- 准备 nnUNet 推理输入：把 test_images/*.tif 组织成 *_0000.tif，并写入 *_0000.json(spacing)
- 调用 nnUNetv2_predict 进行批量推理
- 将 nnUNet 输出（npz/tif/nii.gz）转换为比赛需要的二值 TIFF（0/1）
- （可选）对 surface 概率图做后处理（3D hysteresis + closing + 去小连通域）
- 打包为 submission.zip

前提：
- 你已经训练好了 nnUNet 模型，且 nnUNet_results / nnUNet_preprocessed 指向训练时使用的目录

示例：
  python infer.py \
	--root-dir /data/vesuvius-challenge-surface-detection \
	--work-dir ./work \
	--dataset-id 100 \
	--configuration 3d_fullres \
	--plans-name nnUNetResEncUNetMPlans \
	--trainer nnUNetTrainer_250epochs \
	--fold all \
	--checkpoint checkpoint_final.pth
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import zipfile
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import tifffile


def _try_import_postprocess_deps() -> tuple[object, object]:
	"""延迟导入：只有启用后处理时才需要 scipy/skimage。"""
	try:
		import scipy.ndimage as ndi  # type: ignore
	except Exception as exc:  # pragma: no cover
		raise RuntimeError(
			"启用后处理需要 scipy。请先安装：pip install scipy"
		) from exc
	try:
		from skimage.morphology import remove_small_objects  # type: ignore
	except Exception as exc:  # pragma: no cover
		raise RuntimeError(
			"启用后处理需要 scikit-image。请先安装：pip install scikit-image"
		) from exc
	return ndi, remove_small_objects


def build_anisotropic_struct(z_radius: int, xy_radius: int) -> Optional[np.ndarray]:
	"""构造 3D 各向异性结构元素（用于 closing）。"""
	z, r = int(z_radius), int(xy_radius)
	if z == 0 and r == 0:
		return None
	if z == 0 and r > 0:
		size = 2 * r + 1
		struct = np.zeros((1, size, size), dtype=bool)
		cy, cx = r, r
		for dy in range(-r, r + 1):
			for dx in range(-r, r + 1):
				if dy * dy + dx * dx <= r * r:
					struct[0, cy + dy, cx + dx] = True
		return struct
	if z > 0 and r == 0:
		struct = np.zeros((2 * z + 1, 1, 1), dtype=bool)
		struct[:, 0, 0] = True
		return struct
	depth = 2 * z + 1
	size = 2 * r + 1
	struct = np.zeros((depth, size, size), dtype=bool)
	cz, cy, cx = z, r, r
	for dz in range(-z, z + 1):
		for dy in range(-r, r + 1):
			for dx in range(-r, r + 1):
				if dy * dy + dx * dx <= r * r:
					struct[cz + dz, cy + dy, cx + dx] = True
	return struct


def topo_postprocess(
	probs: np.ndarray,
	*,
	T_low: float = 0.30,
	T_high: float = 0.80,
	z_radius: int = 3,
	xy_radius: int = 2,
	dust_min_size: int = 100,
) -> np.ndarray:
	"""对 3D 概率图做拓扑/形态学后处理，输出 uint8 二值 mask。"""
	ndi, remove_small_objects = _try_import_postprocess_deps()

	# Step 1: 3D Hysteresis
	strong = probs >= T_high
	weak = probs >= T_low
	if not np.any(strong):
		return np.zeros_like(probs, dtype=np.uint8)

	struct_hyst = ndi.generate_binary_structure(3, 3)
	mask = ndi.binary_propagation(strong, mask=weak, structure=struct_hyst)
	if not np.any(mask):
		return np.zeros_like(probs, dtype=np.uint8)

	# Step 2: 3D Anisotropic Closing
	if z_radius > 0 or xy_radius > 0:
		struct_close = build_anisotropic_struct(z_radius, xy_radius)
		if struct_close is not None:
			mask = ndi.binary_closing(mask, structure=struct_close)

	# Step 3: Dust Removal
	if dust_min_size > 0:
		mask = remove_small_objects(mask.astype(bool), min_size=int(dust_min_size))

	return mask.astype(np.uint8)


def run(cmd: str) -> None:
	print(f"Running: {cmd}")
	completed = subprocess.run(cmd, shell=True)
	if completed.returncode != 0:
		raise RuntimeError(f"Command failed (exit={completed.returncode}): {cmd}")


def safe_symlink_or_copy(src: Path, dst: Path) -> None:
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


def prepare_test_input(
	*,
	test_images_dir: Path,
	out_dir: Path,
	spacing: Sequence[float],
	use_symlinks: bool,
) -> Path:
	out_dir.mkdir(parents=True, exist_ok=True)
	tif_paths = sorted(test_images_dir.glob("*.tif"))
	if not tif_paths:
		raise FileNotFoundError(f"No .tif under: {test_images_dir}")

	for src in tif_paths:
		case_id = src.stem
		dst_tif = out_dir / f"{case_id}_0000.tif"
		dst_json = out_dir / f"{case_id}_0000.json"
		if use_symlinks:
			safe_symlink_or_copy(src, dst_tif)
		else:
			if not dst_tif.exists():
				shutil.copy2(src, dst_tif)
		if not dst_json.exists():
			write_spacing_json(dst_json, spacing)
	return out_dir


def run_nnunet_predict(
	*,
	input_dir: Path,
	output_dir: Path,
	dataset_id: int,
	configuration: str,
	fold: str,
	plans_name: str,
	trainer: str,
	checkpoint: str,
	disable_tta: bool,
	save_probabilities: bool,
) -> None:
	output_dir.mkdir(parents=True, exist_ok=True)
	cmd = (
		f"nnUNetv2_predict "
		f"-i \"{input_dir}\" -o \"{output_dir}\" "
		f"-d {dataset_id:03d} -c {configuration} -f {fold} "
		f"-p {plans_name} -tr {trainer} -chk {checkpoint} "
		f"--verbose"
	)
	if disable_tta:
		cmd += " --disable_tta"
	if save_probabilities:
		cmd += " --save_probabilities"
	run(cmd)


def _load_labels_from_pred_file(path: Path) -> np.ndarray:
	if path.suffix == ".npz":
		data = np.load(path)
		probs = data["probabilities"]  # (C, D, H, W)
		return probs.argmax(0).astype(np.uint8)
	if path.suffix == ".tif":
		return tifffile.imread(str(path)).astype(np.uint8)
	if path.name.endswith(".nii.gz") or path.suffix == ".nii":
		# 避免额外依赖：如果你的输出是 NIfTI，请安装 nibabel 后再启用
		raise RuntimeError(
			"Prediction output is NIfTI (.nii/.nii.gz). Install nibabel and extend loader, "
			"or enable --save-probabilities so nnUNet writes .npz."
		)
	raise ValueError(f"Unsupported prediction file: {path}")


def _load_probabilities_from_npz(path: Path) -> np.ndarray:
	data = np.load(path)
	probs = data["probabilities"]
	if probs.ndim != 4:
		raise ValueError(f"Unexpected probabilities shape in {path}: {probs.shape}")
	return probs


def convert_predictions_to_submission_masks(
	*,
	pred_dir: Path,
	out_dir: Path,
	surface_label: int,
	prefer_npz: bool,
	postprocess: bool = False,
	T_low: float = 0.30,
	T_high: float = 0.80,
	z_radius: int = 3,
	xy_radius: int = 2,
	dust_min_size: int = 100,
) -> Path:
	out_dir.mkdir(parents=True, exist_ok=True)

	# 启用后处理时，必须优先使用 .npz 概率图
	if postprocess and not prefer_npz:
		raise ValueError("postprocess=True 时请同时启用 --save-probabilities（生成 .npz）")

	pred_files: list[Path] = []
	if prefer_npz:
		pred_files = sorted(pred_dir.glob("*.npz"))
	if not pred_files:
		pred_files = sorted(pred_dir.glob("*.tif"))
	if not pred_files:
		pred_files = sorted(pred_dir.glob("*.nii.gz"))
	if not pred_files:
		raise FileNotFoundError(f"No prediction files found in: {pred_dir}")

	for p in pred_files:
		case_id = p.stem
		if case_id.endswith(".nii"):
			case_id = case_id[:-4]

		if postprocess:
			if p.suffix != ".npz":
				raise RuntimeError(
					"启用后处理需要 .npz 概率图输出。请运行 nnUNetv2_predict 时加 --save_probabilities。"
				)
			probs = _load_probabilities_from_npz(p)
			if surface_label < 0 or surface_label >= probs.shape[0]:
				raise ValueError(
					f"surface_label={surface_label} 越界，模型类别数为 {probs.shape[0]}"
				)
			surface_probs = probs[int(surface_label)]
			mask = topo_postprocess(
				surface_probs,
				T_low=float(T_low),
				T_high=float(T_high),
				z_radius=int(z_radius),
				xy_radius=int(xy_radius),
				dust_min_size=int(dust_min_size),
			)
		else:
			labels = _load_labels_from_pred_file(p)
			mask = (labels == surface_label).astype(np.uint8)

		tifffile.imwrite(str(out_dir / f"{case_id}.tif"), mask)
	return out_dir


def write_submission_zip(*, test_csv: Path, masks_dir: Path, zip_path: Path) -> Path:
	df = pd.read_csv(str(test_csv))
	zip_path.parent.mkdir(parents=True, exist_ok=True)
	with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
		for image_id in df["id"]:
			tif_path = masks_dir / f"{image_id}.tif"
			if not tif_path.exists():
				raise FileNotFoundError(f"Missing mask: {tif_path}")
			z.write(tif_path, arcname=f"{image_id}.tif")
	return zip_path


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
	p = argparse.ArgumentParser(description="nnUNetv2 inference + submission.zip for Vesuvius")
	p.add_argument("--root-dir", type=Path, required=True, help="比赛数据根目录（包含 test_images/test.csv）")
	p.add_argument("--work-dir", type=Path, default=Path("./work"), help="工作目录（默认 ./work）")

	# nnUNet env
	p.add_argument("--nnunet-raw", type=Path, default=None)
	p.add_argument("--nnunet-preprocessed", type=Path, default=None)
	p.add_argument("--nnunet-results", type=Path, default=None)

	# model config
	p.add_argument("--dataset-id", type=int, default=100)
	p.add_argument("--configuration", type=str, default="3d_fullres")
	p.add_argument("--plans-name", type=str, default="nnUNetResEncUNetMPlans")
	p.add_argument("--trainer", type=str, default="nnUNetTrainer")
	p.add_argument("--fold", type=str, default="all")
	p.add_argument("--checkpoint", type=str, default="checkpoint_final.pth")

	p.add_argument("--disable-tta", type=int, default=0, choices=[0, 1])
	p.add_argument("--save-probabilities", type=int, default=1, choices=[0, 1])
	p.add_argument("--surface-label", type=int, default=1)

	# 后处理（可选）：基于 surface 概率图进行 hysteresis/closing/去小连通域
	p.add_argument("--postprocess", type=int, default=0, choices=[0, 1])
	p.add_argument("--T-low", dest="T_low", type=float, default=0.30)
	p.add_argument("--T-high", dest="T_high", type=float, default=0.80)
	p.add_argument("--z-radius", type=int, default=3)
	p.add_argument("--xy-radius", type=int, default=2)
	p.add_argument("--dust-min-size", type=int, default=100)

	p.add_argument("--spacing", type=float, nargs=3, default=(1.0, 1.0, 1.0))
	p.add_argument("--no-symlinks", action="store_true")

	p.add_argument("--zip-path", type=Path, default=None, help="输出 submission.zip 路径（默认 work-dir/submission.zip）")
	return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
	args = parse_args(argv)
	work_dir: Path = args.work_dir
	work_dir.mkdir(parents=True, exist_ok=True)

	nnunet_raw = args.nnunet_raw or (work_dir / "nnUNet_raw")
	nnunet_preprocessed = args.nnunet_preprocessed or (work_dir / "nnUNet_preprocessed")
	nnunet_results = args.nnunet_results or (work_dir / "nnUNet_results")

	nnunet_raw.mkdir(parents=True, exist_ok=True)
	nnunet_preprocessed.mkdir(parents=True, exist_ok=True)
	nnunet_results.mkdir(parents=True, exist_ok=True)
	os.environ["nnUNet_raw"] = str(nnunet_raw)
	os.environ["nnUNet_preprocessed"] = str(nnunet_preprocessed)
	os.environ["nnUNet_results"] = str(nnunet_results)

	test_images_dir = args.root_dir / "test_images"
	test_csv = args.root_dir / "test.csv"
	if not test_images_dir.exists():
		raise FileNotFoundError(f"test_images not found: {test_images_dir}")
	if not test_csv.exists():
		raise FileNotFoundError(f"test.csv not found: {test_csv}")

	nnunet_input = work_dir / "nnunet_test_input"
	nnunet_pred = work_dir / "nnunet_predictions_raw"
	masks_dir = work_dir / "submission_masks"
	zip_path = args.zip_path or (work_dir / "submission.zip")

	prepare_test_input(
		test_images_dir=test_images_dir,
		out_dir=nnunet_input,
		spacing=args.spacing,
		use_symlinks=not args.no_symlinks,
	)

	run_nnunet_predict(
		input_dir=nnunet_input,
		output_dir=nnunet_pred,
		dataset_id=args.dataset_id,
		configuration=args.configuration,
		fold=args.fold,
		plans_name=args.plans_name,
		trainer=args.trainer,
		checkpoint=args.checkpoint,
		disable_tta=bool(args.disable_tta),
		save_probabilities=bool(args.save_probabilities),
	)

	convert_predictions_to_submission_masks(
		pred_dir=nnunet_pred,
		out_dir=masks_dir,
		surface_label=args.surface_label,
		prefer_npz=bool(args.save_probabilities),
		postprocess=bool(args.postprocess),
		T_low=float(args.T_low),
		T_high=float(args.T_high),
		z_radius=int(args.z_radius),
		xy_radius=int(args.xy_radius),
		dust_min_size=int(args.dust_min_size),
	)

	out_zip = write_submission_zip(test_csv=test_csv, masks_dir=masks_dir, zip_path=zip_path)
	print("Submission ZIP:", out_zip)
	return 0


if __name__ == "__main__":
	raise SystemExit(main())

