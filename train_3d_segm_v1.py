"""3D-SegM（MONAI + PyTorch Lightning）训练脚本：Vesuvius Challenge Surface Detection。

该脚本把 notebook `surface-train-inference-3d-segm-gpu-augment.ipynb` 里的核心流程脚本化：
- 加载 3D 体数据（支持 .tif / .npy / .npz）
- Lightning DataModule（可选：把 resize+增强放到 GPU 上做）
- MONAI 网络（默认 SegResNet）
- 训练 + 保存 ckpt（监控 val_fg_dice：只算前景 class=1）

说明：
- 标签里可能含有 ignore 类（默认 2），loss 与指标会忽略该类。
- 这是 baseline/工程化脚本，便于本地或服务器运行。

本版（v1）关键改动：
- **随机 patch 训练**：默认 patch_size=(192,192,192)，不做 overlap 训练。
- **pos/neg 引导采样**：尽量采到前景/标注区域，避免采到全是 ignore=2 的 patch。
- **严格 ignore mask**：loss/指标只在 valid voxels 上统计，ignore 不进入分母。
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.distributed as dist
from monai.networks.nets import SegResNet, SwinUNETR
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from torch.utils.data import DataLoader, Dataset

import tifffile


def _list_volumes(images_dir: Path) -> list[str]:
	exts = [".npy", ".npz", ".tif"]
	files: list[str] = []
	for ext in exts:
		files.extend([p.name for p in sorted(images_dir.glob(f"*{ext}"))])
	return files


def _load_any(path: Path) -> np.ndarray:
	if path.suffix == ".npy":
		return np.load(str(path))
	if path.suffix == ".npz":
		data = np.load(str(path))
		return data[list(data.files)[0]]
	return tifffile.imread(str(path))


class SurfaceDataset3D(Dataset):
	"""3D 表面检测随机 patch 数据集（pos/neg 引导）。

	- 支持：.tif / .npy / .npz
	- 训练/验证都返回 patch：image (1,d,h,w) float16、mask (1,d,h,w) long、frag_id
	"""

	def __init__(
		self,
		*,
		images_dir: Path,
		labels_dir: Optional[Path],
		volume_files: Optional[list[str]] = None,
		intensity_divisor: float = 255.0,
		ignore_index_val: int = 2,
		patch_size: Tuple[int, int, int] = (192, 192, 192),
		samples_per_volume: int = 16,
		pos_fraction: float = 0.7,
		max_sample_attempts: int = 20,
		augment: bool = True,
		seed: int = 42,
	) -> None:
		super().__init__()
		self.images_dir = images_dir
		self.labels_dir = labels_dir
		self.volume_files = volume_files or _list_volumes(images_dir)
		self.intensity_divisor = float(intensity_divisor)
		self.ignore_index_val = int(ignore_index_val)
		self.patch_size = tuple(int(x) for x in patch_size)
		self.samples_per_volume = int(samples_per_volume)
		self.pos_fraction = float(pos_fraction)
		self.max_sample_attempts = int(max_sample_attempts)
		self.augment = bool(augment)
		self.seed = int(seed)

		if self.samples_per_volume <= 0:
			raise ValueError("samples_per_volume must be >= 1")
		if not (0.0 <= self.pos_fraction <= 1.0):
			raise ValueError("pos_fraction must be in [0, 1]")

		if not self.volume_files:
			raise RuntimeError(f"未找到任何体数据文件：{images_dir}")

		valid_files: list[str] = []
		for filename in self.volume_files:
			img_path = self.images_dir / filename
			if not img_path.exists():
				continue
			if self.labels_dir is not None:
				lbl_path = self.labels_dir / filename
				if not lbl_path.exists():
					continue
			valid_files.append(filename)
		if not valid_files:
			raise RuntimeError(
				"未找到可用的训练样本（请检查 images_dir/labels_dir 文件是否一一对应）。"
			)
		self.volume_files = valid_files

	def __len__(self) -> int:
		# 每个 volume 在一个 epoch 里采样多个 patch
		return len(self.volume_files) * int(self.samples_per_volume)

	@staticmethod
	def _extract_patch(arr: np.ndarray, start: tuple[int, int, int], size: tuple[int, int, int]) -> np.ndarray:
		z0, y0, x0 = start
		dz, dy, dx = size
		return arr[z0 : z0 + dz, y0 : y0 + dy, x0 : x0 + dx]

	@staticmethod
	def _compute_start(center: tuple[int, int, int], size: tuple[int, int, int], shape: tuple[int, int, int]) -> tuple[int, int, int]:
		cz, cy, cx = center
		dz, dy, dx = size
		D, H, W = shape
		z0 = int(cz - dz // 2)
		y0 = int(cy - dy // 2)
		x0 = int(cx - dx // 2)
		z0 = max(0, min(z0, max(0, D - dz)))
		y0 = max(0, min(y0, max(0, H - dy)))
		x0 = max(0, min(x0, max(0, W - dx)))
		return z0, y0, x0

	def _sample_center(self, mask: np.ndarray, *, rng: np.random.Generator) -> tuple[int, int, int]:
		# mask: (D,H,W) with values {0,1,ignore}
		ignore = int(self.ignore_index_val)
		valid = mask != ignore
		fg = mask == 1
		bg = (mask == 0) & valid

		choose_pos = (rng.random() < float(self.pos_fraction)) and np.any(fg)
		if choose_pos:
			coords = np.argwhere(fg)
		elif np.any(bg):
			coords = np.argwhere(bg)
		elif np.any(valid):
			coords = np.argwhere(valid)
		else:
			# 极端情况：全是 ignore
			D, H, W = mask.shape
			return int(rng.integers(0, D)), int(rng.integers(0, H)), int(rng.integers(0, W))

		pick = coords[int(rng.integers(0, len(coords)))]
		return int(pick[0]), int(pick[1]), int(pick[2])

	def __getitem__(self, idx: int):
		# idx 映射到具体 volume；同一个 volume 会被采样多次
		vol_idx = int(idx) % len(self.volume_files)
		filename = self.volume_files[vol_idx]
		image_path = self.images_dir / filename
		image = _load_any(image_path)

		mask: Optional[np.ndarray] = None
		if self.labels_dir is not None:
			mask_path = self.labels_dir / filename
			if mask_path.exists():
				mask = _load_any(mask_path)

		if mask is None:
			raise RuntimeError("训练集必须提供 labels_dir")

		# 若 volume 小于 patch_size，则先 padding 到至少 patch_size（避免返回变长 patch）
		dz, dy, dx = self.patch_size
		D, H, W = image.shape
		pad_z = max(0, dz - D)
		pad_y = max(0, dy - H)
		pad_x = max(0, dx - W)
		if pad_z > 0 or pad_y > 0 or pad_x > 0:
			pad_width = (
				(0, pad_z),
				(0, pad_y),
				(0, pad_x),
			)
			image = np.pad(image, pad_width, mode="constant", constant_values=0)
			mask = np.pad(
				mask,
				pad_width,
				mode="constant",
				constant_values=int(self.ignore_index_val),
			)

		# patch sampling（避免全 ignore patch）
		rng = np.random.default_rng(int(self.seed) + int(idx))
		patch_size = self.patch_size
		for _ in range(max(1, int(self.max_sample_attempts))):
			center = self._sample_center(mask, rng=rng)
			start = self._compute_start(center, patch_size, image.shape)
			img_p = self._extract_patch(image, start, patch_size)
			msk_p = self._extract_patch(mask, start, patch_size)
			if np.any(msk_p != int(self.ignore_index_val)):
				break
		else:
			# 实在采不到 valid，就返回任意 patch
			center = self._sample_center(mask, rng=rng)
			start = self._compute_start(center, patch_size, image.shape)
			img_p = self._extract_patch(image, start, patch_size)
			msk_p = self._extract_patch(mask, start, patch_size)

		# image: (d,h,w) -> Tensor (1,d,h,w)
		image_t = torch.from_numpy(img_p).half().div_(self.intensity_divisor).unsqueeze(0)
		mask_t = torch.from_numpy(msk_p).long().unsqueeze(0)

		# 轻量增强：随机翻转（训练用；验证禁用）
		if self.augment:
			if bool(rng.integers(0, 2)):
				image_t = torch.flip(image_t, dims=[1])
				mask_t = torch.flip(mask_t, dims=[1])
			if bool(rng.integers(0, 2)):
				image_t = torch.flip(image_t, dims=[2])
				mask_t = torch.flip(mask_t, dims=[2])
			if bool(rng.integers(0, 2)):
				image_t = torch.flip(image_t, dims=[3])
				mask_t = torch.flip(mask_t, dims=[3])

		frag_id = Path(filename).stem
		return image_t, mask_t, frag_id


class SurfaceDataModule(pl.LightningDataModule):
	"""Lightning DataModule：负责数据划分与 DataLoader 构建。

	本版使用随机 patch 数据集，DataLoader 直接返回 (B,1,d,h,w)。
	"""

	def __init__(
		self,
		*,
		train_images_dir: Path,
		train_labels_dir: Path,
		patch_size: Tuple[int, int, int],
		val_split: float,
		batch_size: int,
		num_workers: int,
		intensity_divisor: float,
		ignore_index_val: int,
		samples_per_volume: int,
		pos_fraction: float,
		max_sample_attempts: int,
		seed: int,
	) -> None:
		super().__init__()
		self.train_images_dir = train_images_dir
		self.train_labels_dir = train_labels_dir
		self.patch_size = tuple(int(x) for x in patch_size)
		self.val_split = float(val_split)
		self.batch_size = int(batch_size)
		self.num_workers = int(num_workers)
		self.intensity_divisor = float(intensity_divisor)
		self.ignore_index_val = int(ignore_index_val)
		self.samples_per_volume = int(samples_per_volume)
		self.pos_fraction = float(pos_fraction)
		self.max_sample_attempts = int(max_sample_attempts)
		self.seed = int(seed)

		self.train_dataset: Optional[SurfaceDataset3D] = None
		self.val_dataset: Optional[SurfaceDataset3D] = None

		# 本版不再做 resize 到固定体积；模型输入由 patch_size 控制

	def setup(self, stage: Optional[str] = None):
		files = _list_volumes(self.train_images_dir)
		if not files:
			raise RuntimeError(f"train_images_dir 为空：{self.train_images_dir}")

		# 只保留标签存在的样本（避免训练时才报错）
		paired: list[str] = []
		for f in files:
			if (self.train_labels_dir / f).exists():
				paired.append(f)
		if not paired:
			raise RuntimeError(
				f"未找到成对数据：{self.train_images_dir} 与 {self.train_labels_dir}"
			)

		rng = np.random.RandomState(42)
		rng.shuffle(paired)
		split_idx = int(len(paired) * (1 - self.val_split))
		train_files = paired[:split_idx]
		val_files = paired[split_idx:]

		self.train_dataset = SurfaceDataset3D(
			images_dir=self.train_images_dir,
			labels_dir=self.train_labels_dir,
			volume_files=train_files,
			intensity_divisor=self.intensity_divisor,
			ignore_index_val=self.ignore_index_val,
			patch_size=self.patch_size,
			samples_per_volume=self.samples_per_volume,
			pos_fraction=self.pos_fraction,
			max_sample_attempts=self.max_sample_attempts,
			augment=True,
			seed=self.seed,
		)
		self.val_dataset = SurfaceDataset3D(
			images_dir=self.train_images_dir,
			labels_dir=self.train_labels_dir,
			volume_files=val_files,
			intensity_divisor=self.intensity_divisor,
			ignore_index_val=self.ignore_index_val,
			patch_size=self.patch_size,
			samples_per_volume=max(1, self.samples_per_volume // 2),
			pos_fraction=self.pos_fraction,
			max_sample_attempts=self.max_sample_attempts,
			augment=False,
			seed=self.seed + 1337,
		)

	def train_dataloader(self) -> DataLoader:
		assert self.train_dataset is not None
		return DataLoader(
			self.train_dataset,
			batch_size=self.batch_size,
			shuffle=True,
			num_workers=self.num_workers,
			pin_memory=True,
			persistent_workers=bool(self.num_workers > 0),
		)

	def val_dataloader(self) -> DataLoader:
		assert self.val_dataset is not None
		return DataLoader(
			self.val_dataset,
			batch_size=self.batch_size,
			shuffle=False,
			num_workers=self.num_workers,
			pin_memory=True,
			persistent_workers=bool(self.num_workers > 0),
		)


class SurfaceSegmentation3D(pl.LightningModule):
	"""3D 表面分割 LightningModule。"""

	def __init__(
		self,
		*,
		net: nn.Module,
		out_channels: int = 2,
		learning_rate: float = 2e-3,
		weight_decay: float = 1e-4,
		ignore_index_val: int = 2,
		ce_weight: float = 1.0,
		dice_weight: float = 1.0,
		tversky_weight: float = 0.5,
		tversky_alpha: float = 0.7,
		tversky_beta: float = 0.3,
		ema: bool = True,
		ema_decay: float = 0.999,
		ema_start_step: int = 0,
		ema_update_every: int = 1,
	) -> None:
		super().__init__()
		self.save_hyperparameters(ignore=["net"])
		self.net_module = net
		self.ignore_index_val = int(ignore_index_val)

		self._ema_enabled = bool(ema)
		self._ema_state: dict[str, torch.Tensor] = {}
		self._ema_backup: dict[str, torch.Tensor] = {}
		self._ema_last_step: int = -1
		self._ema_in_validation: bool = False

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return self.net_module(x)

	def _ema_init_if_needed(self) -> None:
		if not self._ema_enabled:
			return
		if self._ema_state:
			return
		# 用 fp32 保存 EMA，数值更稳定
		for name, p in self.net_module.named_parameters():
			if not p.requires_grad:
				continue
			self._ema_state[name] = p.detach().float().clone()

	def _ema_update(self) -> None:
		if not self._ema_enabled:
			return
		self._ema_init_if_needed()
		decay = float(self.hparams.ema_decay)
		with torch.no_grad():
			for name, p in self.net_module.named_parameters():
				if not p.requires_grad:
					continue
				if name not in self._ema_state:
					self._ema_state[name] = p.detach().float().clone()
					continue
				self._ema_state[name].mul_(decay).add_(p.detach().float(), alpha=1.0 - decay)

	def _ema_apply(self) -> None:
		if not self._ema_enabled:
			return
		self._ema_init_if_needed()
		if not self._ema_state:
			return
		if self._ema_backup:
			return
		with torch.no_grad():
			for name, p in self.net_module.named_parameters():
				if not p.requires_grad:
					continue
				self._ema_backup[name] = p.detach().clone()
				ema = self._ema_state.get(name)
				if ema is None:
					continue
				p.copy_(ema.to(dtype=p.dtype, device=p.device))

	def _ema_restore(self) -> None:
		if not self._ema_backup:
			return
		with torch.no_grad():
			for name, p in self.net_module.named_parameters():
				bak = self._ema_backup.get(name)
				if bak is None:
					continue
				p.copy_(bak.to(dtype=p.dtype, device=p.device))
		self._ema_backup.clear()

	def on_fit_start(self) -> None:
		# 初始化 EMA（如果启用）
		self._ema_init_if_needed()

	def on_train_batch_end(self, outputs, batch, batch_idx: int) -> None:
		# global_step 在每次 optimizer.step() 后才会增长；用它避免在 gradient accumulation 的“非 step batch”里更新 EMA。
		if not self._ema_enabled:
			return
		gs = int(getattr(self, "global_step", 0))
		if gs <= int(self.hparams.ema_start_step):
			return
		if gs == self._ema_last_step:
			return
		if int(self.hparams.ema_update_every) > 1 and (gs % int(self.hparams.ema_update_every) != 0):
			return
		self._ema_update()
		self._ema_last_step = gs

	def on_validation_epoch_start(self) -> None:
		# 用 EMA 权重做验证（并让 ckpt/earlystop 监控的是 EMA 表现）
		if self._ema_enabled:
			self._ema_apply()
			self._ema_in_validation = True

	def on_validation_epoch_end(self) -> None:
		if self._ema_in_validation:
			self._ema_restore()
			self._ema_in_validation = False

	def on_save_checkpoint(self, checkpoint: dict) -> None:
		# 让保存出来的 checkpoint 权重就是 EMA 权重（满足“保存 best 用 EMA”）
		if not self._ema_enabled:
			return
		self._ema_init_if_needed()
		if not self._ema_state:
			return
		checkpoint["ema_state"] = {k: v.detach().cpu() for k, v in self._ema_state.items()}
		state_dict = checkpoint.get("state_dict")
		if not isinstance(state_dict, dict):
			return
		for name, ema in self._ema_state.items():
			key = f"net_module.{name}"
			if key in state_dict:
				state_dict[key] = ema.to(dtype=state_dict[key].dtype, device=state_dict[key].device)

	def on_load_checkpoint(self, checkpoint: dict) -> None:
		if not self._ema_enabled:
			return
		ema_state = checkpoint.get("ema_state")
		if isinstance(ema_state, dict) and ema_state:
			self._ema_state = {str(k): v.detach().float().clone() for k, v in ema_state.items()}
		else:
			self._ema_state.clear()

	def _masked_ce(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
		# targets: (B,1,D,H,W) with ignore_index
		t = targets.squeeze(1).long()
		return torch.nn.functional.cross_entropy(logits, t, ignore_index=int(self.ignore_index_val))

	def _soft_dice_loss_fg(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
		# foreground-only soft dice, strictly masked by valid voxels
		probs = torch.softmax(logits, dim=1)[:, 1]  # (B,D,H,W)
		gt = (targets.squeeze(1) == 1).float()
		valid = (targets.squeeze(1) != int(self.ignore_index_val)).float()
		p = probs * valid
		g = gt * valid
		intersection = (p * g).sum()
		denom = p.sum() + g.sum()
		dice = (2.0 * intersection + 1e-8) / (denom + 1e-8)
		return 1.0 - dice

	def _tversky_loss_fg(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
		probs = torch.softmax(logits, dim=1)[:, 1]
		gt = (targets.squeeze(1) == 1).float()
		valid = (targets.squeeze(1) != int(self.ignore_index_val)).float()
		p = probs * valid
		g = gt * valid
		tp = (p * g).sum()
		fp = (p * (1.0 - g)).sum()
		fn = ((1.0 - p) * g).sum()
		alpha = float(self.hparams.tversky_alpha)
		beta = float(self.hparams.tversky_beta)
		tv = (tp + 1e-8) / (tp + alpha * fp + beta * fn + 1e-8)
		return 1.0 - tv

	def _compute_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
		loss_ce = self._masked_ce(logits, targets)
		loss_dice = self._soft_dice_loss_fg(logits, targets)
		loss_tv = self._tversky_loss_fg(logits, targets)
		return (
			float(self.hparams.ce_weight) * loss_ce
			+ float(self.hparams.dice_weight) * loss_dice
			+ float(self.hparams.tversky_weight) * loss_tv
		)

	def _fg_dice_hard(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
		preds = torch.argmax(logits, dim=1)  # (B,D,H,W)
		pred_fg = (preds == 1).float()
		gt_fg = (targets.squeeze(1) == 1).float()
		valid = (targets.squeeze(1) != int(self.ignore_index_val)).float()
		intersection = (pred_fg * gt_fg * valid).sum()
		pred_sum = (pred_fg * valid).sum()
		gt_sum = (gt_fg * valid).sum()
		# DDP 下把分子/分母跨卡求和，得到真正的全局 dice（而不是各卡 dice 的平均）
		if dist.is_available() and dist.is_initialized():
			dist.all_reduce(intersection, op=dist.ReduceOp.SUM)
			dist.all_reduce(pred_sum, op=dist.ReduceOp.SUM)
			dist.all_reduce(gt_sum, op=dist.ReduceOp.SUM)
		denom = pred_sum + gt_sum
		return (2.0 * intersection + 1e-8) / (denom + 1e-8)

	def training_step(self, batch, batch_idx: int):
		inputs, targets, _ = batch
		logits = self(inputs)
		loss = self._compute_loss(logits, targets)
		fg_dice = self._fg_dice_hard(logits, targets)

		# 多卡时 metrics 在 _fg_dice_hard 已做 all_reduce，这里避免再次 sync
		self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=False)
		self.log("train_fg_dice", fg_dice, on_step=True, on_epoch=True, prog_bar=True, sync_dist=False)
		return loss

	def validation_step(self, batch, batch_idx: int):
		inputs, targets, _ = batch
		logits = self(inputs)
		loss = self._compute_loss(logits, targets)
		fg_dice = self._fg_dice_hard(logits, targets)

		self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
		# val_fg_dice 已全局聚合（all_reduce），rank0 记录即可
		self.log(
			"val_fg_dice",
			fg_dice,
			on_step=False,
			on_epoch=True,
			prog_bar=True,
			sync_dist=False,
			rank_zero_only=True,
		)
		return loss

	def configure_optimizers(self):
		optimizer = torch.optim.AdamW(
			self.parameters(),
			lr=float(self.hparams.learning_rate),
			weight_decay=float(self.hparams.weight_decay),
		)
		scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
			optimizer,
			T_max=self.trainer.max_epochs if self.trainer else 20,
			eta_min=1e-6,
		)
		return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}


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


@dataclass(frozen=True)
class TrainConfig:
	train_images_dir: Path
	train_labels_dir: Path
	output_dir: Path
	arch: str
	model_input_size: Tuple[int, int, int]
	samples_per_volume: int
	pos_fraction: float
	max_sample_attempts: int
	ema: bool
	ema_decay: float
	ema_start_step: int
	ema_update_every: int
	in_channels: int
	out_channels: int
	ignore_index: int
	intensity_divisor: float
	val_split: float
	batch_size: int
	num_workers: int
	max_epochs: int
	lr: float
	weight_decay: float
	dropout: float
	accumulate_grad_batches: int
	precision: str
	gpu_transforms: bool
	devices: Union[str, int, Tuple[int, ...]]
	strategy: str
	num_nodes: int
	resume_best: bool
	resume_from: Optional[Path]


def parse_args(argv: Optional[Sequence[str]] = None) -> TrainConfig:
	p = argparse.ArgumentParser(description="3D-SegM 训练脚本（MONAI + Lightning）")

	p.add_argument("--train-images-dir", type=Path, required=True)
	p.add_argument("--train-labels-dir", type=Path, required=True)
	p.add_argument("--output-dir", type=Path, default=Path("./work_3d_segm"))

	p.add_argument("--arch", type=str, default="segresnet", choices=["segresnet", "swinunetr"])
	p.add_argument(
		"--patch-size",
		type=int,
		nargs=3,
		default=(192, 192, 192),
		help="随机采样 patch 的尺寸（默认 192 192 192）；SwinUNETR 的网络输入尺寸也使用该值",
	)
	p.add_argument("--in-channels", type=int, default=1)
	p.add_argument("--out-channels", type=int, default=2)
	p.add_argument("--ignore-index", type=int, default=2)
	p.add_argument("--intensity-divisor", type=float, default=255.0)
	p.add_argument("--val-split", type=float, default=0.2)

	# patch 采样策略
	p.add_argument("--samples-per-volume", type=int, default=16, help="每个 volume 每个 epoch 采样多少个 patch")
	p.add_argument(
		"--pos-fraction",
		type=float,
		default=0.7,
		help="pos/neg 引导采样中，采样到前景中心的概率（默认 0.7）",
	)
	p.add_argument("--max-sample-attempts", type=int, default=20, help="为避免全 ignore patch，最多重采样次数")

	# EMA
	p.add_argument("--ema", type=int, default=1, choices=[0, 1], help="是否启用 EMA（验证/保存 best 用 EMA 权重）")
	p.add_argument("--ema-decay", type=float, default=0.999, help="EMA 衰减系数（越大越平滑）")
	p.add_argument("--ema-start-step", type=int, default=0, help="从第几个 global_step 开始更新 EMA")
	p.add_argument("--ema-update-every", type=int, default=1, help="每隔多少个 global_step 更新一次 EMA")

	p.add_argument("--batch-size", type=int, default=1)
	p.add_argument("--num-workers", type=int, default=2)
	p.add_argument("--max-epochs", type=int, default=20)
	p.add_argument("--lr", type=float, default=2e-3)
	p.add_argument("--weight-decay", type=float, default=1e-4)
	p.add_argument("--dropout", type=float, default=0.2)
	p.add_argument("--accumulate-grad-batches", type=int, default=18)
	p.add_argument("--precision", type=str, default="16-mixed")

	# 多 GPU / 多机参数（Lightning 原生支持）
	p.add_argument(
		"--devices",
		type=str,
		default="auto",
		help="使用哪些设备：auto（默认）/ 1 / 0,1,2,3（逗号分隔）",
	)
	p.add_argument(
		"--strategy",
		type=str,
		default="auto",
		help="分布式策略：auto（默认）/ ddp / ddp_spawn（Windows 常用）",
	)
	p.add_argument("--num-nodes", type=int, default=1, help="多机训练节点数（默认 1）")

	# 该版本已不再把 resize/增强搬到 GPU 上做；保留参数位以免误用时困惑
	p.add_argument("--gpu-transforms", type=int, default=0, choices=[0, 1], help="已弃用：v1 使用 patch 训练")

	p.add_argument(
		"--resume-best",
		type=int,
		default=0,
		choices=[0, 1],
		help="若 output-dir 下已有 ckpt，自动从 val_fg_dice 最好的那个继续训练",
	)
	p.add_argument(
		"--resume-from",
		type=Path,
		default=None,
		help="显式指定某个 ckpt 继续训练（优先级高于 --resume-best）",
	)

	args = p.parse_args(argv)
	devices: Union[str, int, Tuple[int, ...]]
	if str(args.devices).lower() == "auto":
		devices = "auto"
	else:
		dev_s = str(args.devices)
		if "," in dev_s:
			devices = tuple(int(x) for x in dev_s.split(",") if x.strip() != "")
		else:
			devices = int(dev_s)
	return TrainConfig(
		train_images_dir=args.train_images_dir,
		train_labels_dir=args.train_labels_dir,
		output_dir=args.output_dir,
		arch=args.arch,
		model_input_size=tuple(args.patch_size),
		samples_per_volume=int(args.samples_per_volume),
		pos_fraction=float(args.pos_fraction),
		max_sample_attempts=int(args.max_sample_attempts),
		ema=bool(args.ema),
		ema_decay=float(args.ema_decay),
		ema_start_step=int(args.ema_start_step),
		ema_update_every=int(args.ema_update_every),
		in_channels=args.in_channels,
		out_channels=args.out_channels,
		ignore_index=args.ignore_index,
		intensity_divisor=args.intensity_divisor,
		val_split=args.val_split,
		batch_size=args.batch_size,
		num_workers=args.num_workers,
		max_epochs=args.max_epochs,
		lr=args.lr,
		weight_decay=args.weight_decay,
		dropout=args.dropout,
		accumulate_grad_batches=args.accumulate_grad_batches,
		precision=args.precision,
		gpu_transforms=bool(args.gpu_transforms),
		devices=devices,
		strategy=str(args.strategy),
		num_nodes=int(args.num_nodes),
		resume_best=bool(args.resume_best),
		resume_from=args.resume_from,
	)


def main(argv: Optional[Sequence[str]] = None) -> int:
	cfg = parse_args(argv)
	cfg.output_dir.mkdir(parents=True, exist_ok=True)

	# Data
	datamodule = SurfaceDataModule(
		train_images_dir=cfg.train_images_dir,
		train_labels_dir=cfg.train_labels_dir,
		patch_size=cfg.model_input_size,
		val_split=cfg.val_split,
		batch_size=cfg.batch_size,
		num_workers=cfg.num_workers,
		intensity_divisor=cfg.intensity_divisor,
		ignore_index_val=cfg.ignore_index,
		samples_per_volume=cfg.samples_per_volume,
		pos_fraction=cfg.pos_fraction,
		max_sample_attempts=cfg.max_sample_attempts,
		seed=42,
	)
	datamodule.setup()

	# Model
	net = build_net(
		arch=cfg.arch,
		in_channels=cfg.in_channels,
		out_channels=cfg.out_channels,
		model_input_size=cfg.model_input_size,
		dropout=cfg.dropout,
	)
	model = SurfaceSegmentation3D(
		net=net,
		out_channels=cfg.out_channels,
		learning_rate=cfg.lr,
		weight_decay=cfg.weight_decay,
		ignore_index_val=cfg.ignore_index,
		ema=cfg.ema,
		ema_decay=cfg.ema_decay,
		ema_start_step=cfg.ema_start_step,
		ema_update_every=cfg.ema_update_every,
	)

	# Callbacks / Logger
	ckpt_cb = ModelCheckpoint(
		dirpath=cfg.output_dir,
		filename=f"{net.__class__.__name__}" + "-{epoch:02d}-{val_fg_dice:.4f}",
		monitor="val_fg_dice",
		mode="max",
		save_top_k=3,
		verbose=True,
	)
	early_stop = EarlyStopping(
		monitor="val_fg_dice",
		patience=10,
		mode="max",
		verbose=True,
	)
	lr_monitor = LearningRateMonitor(logging_interval="epoch")
	logger = CSVLogger(save_dir=str(cfg.output_dir))

	trainer = pl.Trainer(
		max_epochs=cfg.max_epochs,
		accelerator="auto",
		devices=cfg.devices,
		strategy=("auto" if str(cfg.strategy).lower() == "auto" else cfg.strategy),
		num_nodes=int(cfg.num_nodes),
		logger=logger,
		callbacks=[ckpt_cb, early_stop, lr_monitor],
		precision=cfg.precision,
		log_every_n_steps=1,
		enable_progress_bar=True,
		accumulate_grad_batches=cfg.accumulate_grad_batches,
		gradient_clip_val=1.0,
	)

	ckpt_path: Optional[str] = None
	if cfg.resume_from is not None:
		ckpt_path = str(cfg.resume_from)
	elif cfg.resume_best:
		best_ckpt, best_score = get_best_checkpoint(
			[cfg.output_dir], name_prefix=net.__class__.__name__, metric_name="val_fg_dice"
		)
		if best_ckpt is not None:
			print(f"从 best ckpt 继续训练：{best_ckpt} (val_fg_dice={best_score})")
			ckpt_path = str(best_ckpt)

	try:
		trainer.fit(model, datamodule=datamodule, ckpt_path=ckpt_path)
	except MisconfigurationException as ex:
		# 常见于环境/设备配置问题
		print(ex)
		return 2

	print("训练完成。最优 ckpt：", ckpt_cb.best_model_path)
	return 0


if __name__ == "__main__":
	raise SystemExit(main())

