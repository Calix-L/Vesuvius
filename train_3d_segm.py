"""3D-SegM（MONAI + PyTorch Lightning）训练脚本：Vesuvius Challenge Surface Detection。

该脚本把 notebook `surface-train-inference-3d-segm-gpu-augment.ipynb` 里的核心流程脚本化：
- 加载 3D 体数据（支持 .tif / .npy / .npz）
- Lightning DataModule（可选：把 resize+增强放到 GPU 上做）
- MONAI 网络（默认 SegResNet）
- 训练 + 保存 ckpt（监控 val_dice）

说明：
- 标签里可能含有 ignore 类（默认 2），loss 与指标会忽略该类。
- 这是 baseline/工程化脚本，便于本地或服务器运行。
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
from monai.losses import DiceCELoss, TverskyLoss
from monai import transforms as MT
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
	"""3D 表面检测数据集。

	- 支持：.tif / .npy / .npz
	- 返回：image (C,D,H,W) float16、mask (1,D,H,W) long、frag_id
	"""

	def __init__(
		self,
		*,
		images_dir: Path,
		labels_dir: Optional[Path],
		volume_files: Optional[list[str]] = None,
		intensity_divisor: float = 255.0,
		ignore_index_val: int = 2,
	) -> None:
		super().__init__()
		self.images_dir = images_dir
		self.labels_dir = labels_dir
		self.volume_files = volume_files or _list_volumes(images_dir)
		self.intensity_divisor = float(intensity_divisor)
		self.ignore_index_val = int(ignore_index_val)

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
		return len(self.volume_files)

	def __getitem__(self, idx: int):
		filename = self.volume_files[idx]
		image_path = self.images_dir / filename
		image = _load_any(image_path)

		mask: Optional[np.ndarray] = None
		if self.labels_dir is not None:
			mask_path = self.labels_dir / filename
			if mask_path.exists():
				mask = _load_any(mask_path)

		# image: (D,H,W) -> Tensor (1,D,H,W)
		image_t = torch.from_numpy(image).half().div_(self.intensity_divisor).unsqueeze(0)

		if mask is None:
			# 测试集/无标签：用 ignore_index 填充
			mask_t = torch.full_like(image_t, self.ignore_index_val, dtype=torch.long)
		else:
			mask_t = torch.from_numpy(mask).long().unsqueeze(0)

		frag_id = Path(filename).stem
		return image_t, mask_t, frag_id


def custom_collate(batch):
	"""自定义 collate：返回 list，允许后续在 GPU 上做 resize/增强。"""

	return batch


class SurfaceDataModule(pl.LightningDataModule):
	"""Lightning DataModule：负责数据划分与 DataLoader 构建。

	可选：在 `on_after_batch_transfer` 内把数据搬到 GPU，并执行 MONAI transforms（resize + augment）。
	"""

	def __init__(
		self,
		*,
		train_images_dir: Path,
		train_labels_dir: Path,
		model_input_size: Tuple[int, int, int],
		val_split: float,
		batch_size: int,
		num_workers: int,
		gpu_transforms: bool,
		intensity_divisor: float,
		ignore_index_val: int,
	) -> None:
		super().__init__()
		self.train_images_dir = train_images_dir
		self.train_labels_dir = train_labels_dir
		self.model_input_size = tuple(int(x) for x in model_input_size)
		self.val_split = float(val_split)
		self.batch_size = int(batch_size)
		self.num_workers = int(num_workers)
		self.gpu_transforms = bool(gpu_transforms)
		self.intensity_divisor = float(intensity_divisor)
		self.ignore_index_val = int(ignore_index_val)

		self.train_dataset: Optional[SurfaceDataset3D] = None
		self.val_dataset: Optional[SurfaceDataset3D] = None

		self.gpu_augments = MT.Compose(
			[
				MT.Resized(
					keys=["image", "label"],
					spatial_size=self.model_input_size,
					mode=["trilinear", "nearest"],
				),
				MT.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
				MT.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
				MT.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
				MT.RandRotated(
					keys=["image", "label"],
					range_x=0.1,
					range_y=0.1,
					range_z=0.1,
					prob=0.3,
					keep_size=True,
					mode=["bilinear", "nearest"],
				),
				MT.RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.5),
				MT.RandGaussianNoised(keys=["image"], prob=0.3, mean=0.0, std=0.01),
			]
		)
		self.val_augments = MT.Compose(
			[
				MT.Resized(
					keys=["image", "label"],
					spatial_size=self.model_input_size,
					mode=["trilinear", "nearest"],
				)
			]
		)

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
		)
		self.val_dataset = SurfaceDataset3D(
			images_dir=self.train_images_dir,
			labels_dir=self.train_labels_dir,
			volume_files=val_files,
			intensity_divisor=self.intensity_divisor,
			ignore_index_val=self.ignore_index_val,
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
			collate_fn=custom_collate,
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
			collate_fn=custom_collate,
		)

	def on_after_batch_transfer(self, batch, dataloader_idx):
		if not self.gpu_transforms:
			return batch
		if not isinstance(batch, list):
			return batch

		# Lightning 下更可靠的 root_device
		if self.trainer is not None:
			device = self.trainer.strategy.root_device
			training = self.trainer.training
		else:
			device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
			training = False

		transforms = self.gpu_augments if training else self.val_augments

		x_list, y_list, frag_ids = [], [], []
		for x, y, frag_id in batch:
			x = x.to(device, non_blocking=True)
			y = y.to(device, non_blocking=True)
			data = {"image": x, "label": y}
			data = transforms(data)
			x_list.append(data["image"])
			y_list.append(data["label"])
			frag_ids.append(frag_id)
		return torch.stack(x_list), torch.stack(y_list), frag_ids


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
	) -> None:
		super().__init__()
		self.save_hyperparameters(ignore=["net"])
		self.net_module = net
		self.ignore_index_val = int(ignore_index_val)

		self.criterion_tversky = TverskyLoss(
			softmax=True,
			to_onehot_y=False,
			include_background=True,
			alpha=0.7,
			beta=0.3,
		)
		self.criterion_dice_ce = DiceCELoss(
			softmax=True,
			to_onehot_y=False,
			include_background=True,
		)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return self.net_module(x)

	def _compute_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
		# targets: (B,1,D,H,W) with ignore_index
		valid_mask = (targets != self.ignore_index_val)

		targets_sq = targets.squeeze(1)
		targets_clean = torch.where(
			valid_mask.squeeze(1),
			targets_sq,
			torch.tensor(0, device=targets.device),
		)
		targets_onehot = torch.nn.functional.one_hot(
			targets_clean.long(), num_classes=int(self.hparams.out_channels)
		).float()
		targets_onehot = targets_onehot.permute(0, 4, 1, 2, 3)

		targets_masked = targets_onehot * valid_mask.float()

		loss_tversky = self.criterion_tversky(logits, targets_masked)
		loss_dice_ce = self.criterion_dice_ce(logits, targets_masked)
		return loss_tversky + loss_dice_ce

	@staticmethod
	def _dice_iou_for_class(
		pred: torch.Tensor, target: torch.Tensor, valid: torch.Tensor
	) -> tuple[torch.Tensor, torch.Tensor]:
		# pred/target/valid: (B,1,D,H,W) float/bool
		intersection = (pred * target * valid).sum()
		union_dice = (pred * valid).sum() + (target * valid).sum()
		union_iou = (pred * valid).sum() + (target * valid).sum() - intersection
		dice = (2 * intersection + 1e-8) / (union_dice + 1e-8)
		iou = (intersection + 1e-8) / (union_iou + 1e-8)
		return dice, iou

	def _compute_metrics(self, logits: torch.Tensor, targets: torch.Tensor) -> dict:
		probs = torch.softmax(logits, dim=1)
		preds = torch.argmax(probs, dim=1, keepdim=True)
		valid = (targets != self.ignore_index_val).float()

		dices: list[torch.Tensor] = []
		ious: list[torch.Tensor] = []
		for cls_idx in range(int(self.hparams.out_channels)):
			pred_i = (preds == cls_idx).float()
			targ_i = (targets == cls_idx).float()
			d, j = self._dice_iou_for_class(pred_i, targ_i, valid)
			dices.append(d)
			ious.append(j)
		return {"dice": torch.mean(torch.stack(dices)), "iou": torch.mean(torch.stack(ious))}

	def training_step(self, batch, batch_idx: int):
		inputs, targets, _ = batch
		logits = self(inputs)
		loss = self._compute_loss(logits, targets)
		metrics = self._compute_metrics(logits, targets)

		self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
		self.log("train_dice", metrics["dice"], on_step=True, on_epoch=True, prog_bar=True)
		self.log("train_iou", metrics["iou"], on_step=True, on_epoch=True, prog_bar=True)
		return loss

	def validation_step(self, batch, batch_idx: int):
		inputs, targets, _ = batch
		logits = self(inputs)
		loss = self._compute_loss(logits, targets)
		metrics = self._compute_metrics(logits, targets)

		self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
		self.log("val_dice", metrics["dice"], on_step=False, on_epoch=True, prog_bar=True)
		self.log("val_iou", metrics["iou"], on_step=False, on_epoch=True, prog_bar=True)
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
	metric_name: str = "val_dice",
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
	resume_best: bool
	resume_from: Optional[Path]


def parse_args(argv: Optional[Sequence[str]] = None) -> TrainConfig:
	p = argparse.ArgumentParser(description="3D-SegM 训练脚本（MONAI + Lightning）")

	p.add_argument("--train-images-dir", type=Path, required=True)
	p.add_argument("--train-labels-dir", type=Path, required=True)
	p.add_argument("--output-dir", type=Path, default=Path("./work_3d_segm"))

	p.add_argument("--arch", type=str, default="segresnet", choices=["segresnet", "swinunetr"])
	p.add_argument("--model-input-size", type=int, nargs=3, default=(160, 160, 160))
	p.add_argument("--in-channels", type=int, default=1)
	p.add_argument("--out-channels", type=int, default=2)
	p.add_argument("--ignore-index", type=int, default=2)
	p.add_argument("--intensity-divisor", type=float, default=255.0)
	p.add_argument("--val-split", type=float, default=0.2)

	p.add_argument("--batch-size", type=int, default=1)
	p.add_argument("--num-workers", type=int, default=2)
	p.add_argument("--max-epochs", type=int, default=20)
	p.add_argument("--lr", type=float, default=2e-3)
	p.add_argument("--weight-decay", type=float, default=1e-4)
	p.add_argument("--dropout", type=float, default=0.2)
	p.add_argument("--accumulate-grad-batches", type=int, default=18)
	p.add_argument("--precision", type=str, default="16-mixed")

	p.add_argument("--gpu-transforms", type=int, default=1, choices=[0, 1])

	p.add_argument(
		"--resume-best",
		type=int,
		default=0,
		choices=[0, 1],
		help="若 output-dir 下已有 ckpt，自动从 val_dice 最好的那个继续训练",
	)
	p.add_argument(
		"--resume-from",
		type=Path,
		default=None,
		help="显式指定某个 ckpt 继续训练（优先级高于 --resume-best）",
	)

	args = p.parse_args(argv)
	return TrainConfig(
		train_images_dir=args.train_images_dir,
		train_labels_dir=args.train_labels_dir,
		output_dir=args.output_dir,
		arch=args.arch,
		model_input_size=tuple(args.model_input_size),
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
		model_input_size=cfg.model_input_size,
		val_split=cfg.val_split,
		batch_size=cfg.batch_size,
		num_workers=cfg.num_workers,
		gpu_transforms=cfg.gpu_transforms,
		intensity_divisor=cfg.intensity_divisor,
		ignore_index_val=cfg.ignore_index,
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
	)

	# Callbacks / Logger
	ckpt_cb = ModelCheckpoint(
		dirpath=cfg.output_dir,
		filename=f"{net.__class__.__name__}" + "-{epoch:02d}-{val_dice:.4f}",
		monitor="val_dice",
		mode="max",
		save_top_k=3,
		verbose=True,
	)
	early_stop = EarlyStopping(
		monitor="val_dice",
		patience=10,
		mode="max",
		verbose=True,
	)
	lr_monitor = LearningRateMonitor(logging_interval="epoch")
	logger = CSVLogger(save_dir=str(cfg.output_dir))

	trainer = pl.Trainer(
		max_epochs=cfg.max_epochs,
		accelerator="auto",
		devices="auto",
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
			[cfg.output_dir], name_prefix=net.__class__.__name__
		)
		if best_ckpt is not None:
			print(f"从 best ckpt 继续训练：{best_ckpt} (val_dice={best_score})")
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

