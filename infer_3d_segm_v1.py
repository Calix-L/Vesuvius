"""3D-SegM（MONAI + PyTorch Lightning）推理脚本：生成 submission.zip。

该脚本把 notebook `surface-train-inference-3d-segm-gpu-augment.ipynb` 的推理/打包流程脚本化：
- 加载测试集 3D 体数据（默认 test_images/*.tif）
- 读取训练 ckpt，推理得到 0/1 mask
- 将 mask 保存为 tif（与 test.csv 的 id 对应）
- 打包 submission.zip

注意：
- 推理使用 **原尺寸** sliding window inference，不再做“先 resize 再插回”的流程（避免拓扑硬伤）。
- 可选：复用 predictions notebook 的后处理（3D hysteresis + 各向异性 closing + 去小连通域）。
"""

from __future__ import annotations

import argparse
import contextlib
import re
import zipfile
from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
from monai.inferers import sliding_window_inference
from monai.networks.nets import SegResNet, SwinUNETR

import tifffile


def _try_import_postprocess_deps() -> tuple[object, object]:
	"""延迟导入：只有启用后处理时才需要 scipy/skimage。"""
	try:
		import scipy.ndimage as ndi  # type: ignore
	except Exception as exc:  # pragma: no cover
		raise RuntimeError("启用后处理需要 scipy。请先安装：pip install scipy") from exc
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
	"""predictions notebook 同款后处理：3D hysteresis -> closing -> 去小连通域。

	输入：前景概率图 (D,H,W) float
	输出：二值 mask (D,H,W) uint8
	"""
	ndi, remove_small_objects = _try_import_postprocess_deps()

	# Step 1: 3D Hysteresis
	strong = probs >= float(T_high)
	weak = probs >= float(T_low)
	if not np.any(strong):
		return np.zeros_like(probs, dtype=np.uint8)

	struct_hyst = ndi.generate_binary_structure(3, 3)
	mask = ndi.binary_propagation(strong, mask=weak, structure=struct_hyst)
	if not np.any(mask):
		return np.zeros_like(probs, dtype=np.uint8)

	# Step 2: 3D Anisotropic Closing
	if int(z_radius) > 0 or int(xy_radius) > 0:
		struct_close = build_anisotropic_struct(int(z_radius), int(xy_radius))
		if struct_close is not None:
			mask = ndi.binary_closing(mask, structure=struct_close)

	# Step 3: Dust Removal
	if int(dust_min_size) > 0:
		mask = remove_small_objects(mask.astype(bool), min_size=int(dust_min_size))

	return mask.astype(np.uint8)


def _list_test_cases(test_images_dir: Path) -> list[Path]:
	return sorted(test_images_dir.glob("*.tif"))


def get_best_checkpoint(
	checkpoint_dirs: Union[str, Path, Sequence[Union[str, Path]]],
	*,
	name_prefix: str = "",
	metric_name: str = "val_fg_dice",
) -> tuple[Optional[Path], Optional[float]]:
	if isinstance(checkpoint_dirs, (str, Path)):
		dirs = [Path(checkpoint_dirs)]
	else:
		dirs = [Path(d) for d in checkpoint_dirs]
	dirs = [d for d in dirs if d.exists()]
	if not dirs:
		return None, None

	pattern = re.compile(rf"{re.escape(metric_name)}=?([0-9]+\.[0-9]+)")
	candidates: list[tuple[float, Path]] = []
	for d in dirs:
		for ckpt in d.glob(f"{name_prefix}*.ckpt"):
			m = pattern.search(ckpt.name)
			if not m:
				continue
			candidates.append((float(m.group(1)), ckpt))
	if not candidates:
		return None, None
	candidates.sort(key=lambda x: x[0], reverse=True)
	best_score, best_path = candidates[0]
	return best_path, best_score


class SurfaceSegmentation3D(pl.LightningModule):
	"""推理侧只需要 forward；这里保持与训练脚本一致的 checkpoint 结构。"""

	def __init__(self, *, net: nn.Module) -> None:
		super().__init__()
		self.save_hyperparameters(ignore=["net"])
		self.net_module = net

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return self.net_module(x)


def build_net(
	*,
	arch: str,
	in_channels: int,
	out_channels: int,
	model_input_size: Tuple[int, int, int],
	dropout: float,
) -> nn.Module:
	arch = arch.lower()
	if arch == "segresnet":
		return SegResNet(
			spatial_dims=3,
			in_channels=int(in_channels),
			out_channels=int(out_channels),
			init_filters=16,
			dropout_prob=float(dropout),
		)
	if arch == "swinunetr":
		return SwinUNETR(
			in_channels=int(in_channels),
			out_channels=int(out_channels),
			img_size=tuple(int(x) for x in model_input_size),
			feature_size=48,
			use_v2=True,
			drop_rate=float(dropout),
			attn_drop_rate=float(dropout),
			dropout_path_rate=float(dropout),
		)
	raise ValueError(f"未知 arch：{arch}（可选：segresnet / swinunetr）")


def _autocast_enabled(device: torch.device) -> bool:
	# 简单策略：CUDA 上启用 autocast
	return device.type == "cuda"


def _run_sw_inference(
	*,
	model: pl.LightningModule,
	x: torch.Tensor,
	roi_size: Tuple[int, int, int],
	sw_batch_size: int,
	overlap: float,
	use_amp: bool,
) -> torch.Tensor:
	"""返回 probs: (1,C,D,H,W)"""
	device = x.device
	ctx = (
		torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp)
		if device.type == "cuda"
		else contextlib.nullcontext()
	)
	with torch.no_grad():
		with ctx:
			logits = sliding_window_inference(
				x,
				roi_size=roi_size,
				sw_batch_size=int(sw_batch_size),
				predictor=model,
				overlap=float(overlap),
			)
		probs = torch.softmax(logits, dim=1)
	return probs


def _apply_light_tta(
	*,
	x: torch.Tensor,
	noise_std: float,
	scale_jitter: float,
	seed: int,
) -> torch.Tensor:
	"""轻扰动 TTA：不做翻转，只做一次轻微强度扰动 + 高斯噪声。

	x: (1,1,D,H,W) 取值大致在 [0,1]
	"""
	gen = torch.Generator(device=x.device)
	gen.manual_seed(int(seed))
	x2 = x
	if float(scale_jitter) > 0:
		# scale in [1-scale_jitter, 1+scale_jitter]
		s = (torch.rand((), generator=gen, device=x.device) * 2 - 1) * float(scale_jitter)
		x2 = x2 * (1.0 + s)
	if float(noise_std) > 0:
		noise = torch.randn_like(x2, generator=gen) * float(noise_std)
		x2 = x2 + noise
	# clamp to reasonable range
	return torch.clamp(x2, 0.0, 1.0)


def write_submission_zip(*, test_csv: Path, masks_dir: Path, zip_path: Path) -> Path:
	df = pd.read_csv(str(test_csv))
	zip_path.parent.mkdir(parents=True, exist_ok=True)
	with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
		for image_id in df["id"]:
			tif_path = masks_dir / f"{image_id}.tif"
			if not tif_path.exists():
				raise FileNotFoundError(f"缺少 mask：{tif_path}")
			z.write(tif_path, arcname=f"{image_id}.tif")
	return zip_path


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
	p = argparse.ArgumentParser(description="3D-SegM 推理 + submission.zip")

	p.add_argument("--root-dir", type=Path, required=True, help="比赛数据根目录（包含 test_images/test.csv）")
	p.add_argument("--work-dir", type=Path, default=Path("./work_3d_segm_infer"))

	p.add_argument("--checkpoint", type=Path, default=None, help="要加载的 .ckpt；不传则从 checkpoint-dir 找 best")
	p.add_argument(
		"--checkpoint-dir",
		type=Path,
		default=Path("./work_3d_segm"),
		help="训练输出目录（包含 .ckpt），用于自动寻找 best ckpt",
	)

	p.add_argument("--arch", type=str, default="segresnet", choices=["segresnet", "swinunetr"])
	# SwinUNETR 需要固定 img_size；SegResNet 对输入尺寸更灵活
	p.add_argument(
		"--net-input-size",
		type=int,
		nargs=3,
		default=(192, 192, 192),
		help="构建网络时使用的输入尺寸（训练时的 patch_size；SwinUNETR 必须匹配训练）",
	)
	p.add_argument("--in-channels", type=int, default=1)
	p.add_argument("--out-channels", type=int, default=2)
	p.add_argument("--surface-channel", type=int, default=1, help="前景类别通道索引（默认 1）")
	p.add_argument("--dropout", type=float, default=0.2)

	# Sliding window inference（原尺寸）
	p.add_argument(
		"--roi-size",
		type=int,
		nargs=3,
		default=(192, 192, 192),
		help="滑窗推理的 roi_size（建议 192 或 224；需 <= 体数据尺寸）",
	)
	p.add_argument(
		"--overlap",
		type=float,
		default=0.6,
		help="滑窗 overlap（0.5~0.7 越大越平滑，但更慢/更占显存）",
	)
	p.add_argument("--sw-batch-size", type=int, default=1, help="sliding window 的 batch 数（显存充足可增大）")
	p.add_argument("--amp", type=int, default=1, choices=[0, 1], help="CUDA 上是否启用 autocast")

	# 轻扰动 TTA（两次推理取均值）
	p.add_argument("--tta", type=int, default=0, choices=[0, 1], help="是否启用轻扰动 TTA（两次推理取均值）")
	p.add_argument("--tta-noise-std", type=float, default=0.01, help="TTA 高斯噪声标准差（默认 0.01）")
	p.add_argument("--tta-scale-jitter", type=float, default=0.02, help="TTA 强度缩放抖动幅度（默认 0.02）")
	p.add_argument("--tta-seed", type=int, default=123, help="TTA 随机种子（保证可复现）")

	p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
	p.add_argument("--intensity-divisor", type=float, default=255.0)

	# predictions notebook 风格后处理（可选）
	p.add_argument("--postprocess", type=int, default=0, choices=[0, 1])
	p.add_argument("--T-low", dest="T_low", type=float, default=0.30)
	p.add_argument("--T-high", dest="T_high", type=float, default=0.80)
	p.add_argument("--z-radius", type=int, default=3)
	p.add_argument("--xy-radius", type=int, default=2)
	p.add_argument("--dust-min-size", type=int, default=100)

	p.add_argument("--zip-path", type=Path, default=None)
	return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
	args = parse_args(argv)
	work_dir: Path = args.work_dir
	work_dir.mkdir(parents=True, exist_ok=True)

	test_images_dir = args.root_dir / "test_images"
	test_csv = args.root_dir / "test.csv"
	if not test_images_dir.exists():
		raise FileNotFoundError(f"test_images 不存在：{test_images_dir}")
	if not test_csv.exists():
		raise FileNotFoundError(f"test.csv 不存在：{test_csv}")

	masks_dir = work_dir / "submission_masks"
	masks_dir.mkdir(parents=True, exist_ok=True)
	zip_path = args.zip_path or (work_dir / "submission.zip")

	net_input_size = tuple(int(x) for x in args.net_input_size)
	roi_size = tuple(int(x) for x in args.roi_size)
	overlap = float(args.overlap)
	if not (0.0 <= overlap < 1.0):
		raise ValueError("--overlap must be in [0, 1)")

	# 选择设备
	if args.device == "cpu":
		device = torch.device("cpu")
	elif args.device == "cuda":
		device = torch.device("cuda")
	else:
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	# 选择 ckpt
	ckpt_path: Optional[Path] = args.checkpoint
	net_name = "SegResNet" if args.arch == "segresnet" else "SwinUNETR"
	if ckpt_path is None:
		best, score = get_best_checkpoint(
			[args.checkpoint_dir], name_prefix=net_name, metric_name="val_fg_dice"
		)
		if best is None:
			raise FileNotFoundError(
				f"未指定 --checkpoint，且在 {args.checkpoint_dir} 没找到 {net_name}*.ckpt"
			)
		print(f"自动选择 best ckpt：{best} (val_fg_dice={score})")
		ckpt_path = best

	# 构建网络并加载权重
	net = build_net(
		arch=args.arch,
		in_channels=args.in_channels,
		out_channels=args.out_channels,
		model_input_size=net_input_size,
		dropout=args.dropout,
	)

	# 注意：训练脚本里 save_hyperparameters(ignore=["net"])，所以 load 时必须传 net
	model = SurfaceSegmentation3D.load_from_checkpoint(str(ckpt_path), net=net)
	model.eval()
	model.to(device)

	# 推理
	test_paths = _list_test_cases(test_images_dir)
	if not test_paths:
		raise RuntimeError(f"test_images 目录下没有 .tif：{test_images_dir}")

	for p in test_paths:
		case_id = p.stem
		vol = tifffile.imread(str(p))
		original_shape = tuple(int(x) for x in vol.shape)

		# (D,H,W) -> (1,1,D,H,W)
		x = torch.from_numpy(vol)
		if device.type == "cuda":
			x = x.half()
		else:
			x = x.float()
		x = x.div_(float(args.intensity_divisor)).unsqueeze(0).unsqueeze(0).to(device)

		with torch.no_grad():
			use_amp = bool(args.amp) and _autocast_enabled(device)
			probs1 = _run_sw_inference(
				model=model,
				x=x,
				roi_size=roi_size,
				sw_batch_size=int(args.sw_batch_size),
				overlap=float(overlap),
				use_amp=use_amp,
			)
			if bool(args.tta):
				x_tta = _apply_light_tta(
					x=x,
					noise_std=float(args.tta_noise_std),
					scale_jitter=float(args.tta_scale_jitter),
					seed=int(args.tta_seed),
				)
				probs2 = _run_sw_inference(
					model=model,
					x=x_tta,
					roi_size=roi_size,
					sw_batch_size=int(args.sw_batch_size),
					overlap=float(overlap),
					use_amp=use_amp,
				)
				probs = 0.5 * (probs1 + probs2)
			else:
				probs = probs1

		if bool(args.postprocess):
			surface_ch = int(args.surface_channel)
			if surface_ch < 0 or surface_ch >= probs.shape[1]:
				raise ValueError(
					f"surface-channel={surface_ch} 越界：模型输出通道数为 {probs.shape[1]}"
				)
			surface_probs = probs[0, surface_ch].detach().float().cpu().numpy()
			mask = topo_postprocess(
				surface_probs,
				T_low=float(args.T_low),
				T_high=float(args.T_high),
				z_radius=int(args.z_radius),
				xy_radius=int(args.xy_radius),
				dust_min_size=int(args.dust_min_size),
			)
			pred_bin = torch.from_numpy(mask).to(torch.uint8)
		else:
			pred = torch.argmax(probs, dim=1).squeeze(0)  # (D,H,W)
			pred_bin = (pred == int(args.surface_channel)).to(torch.uint8)

		# 原尺寸输出（无需插值回填）
		if tuple(int(x) for x in pred_bin.shape) != original_shape:
			raise RuntimeError(f"Unexpected output shape: pred={tuple(pred_bin.shape)} vs original={original_shape}")
		tifffile.imwrite(str(masks_dir / f"{case_id}.tif"), pred_bin.cpu().numpy())

	out_zip = write_submission_zip(test_csv=test_csv, masks_dir=masks_dir, zip_path=zip_path)
	print("Submission ZIP:", out_zip)
	return 0


if __name__ == "__main__":
	raise SystemExit(main())

