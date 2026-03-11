"""nnUNetv2 训练脚本（仅训练阶段）

本脚本只负责调用 `nnUNetv2_train` 进行训练/续训。

数据准备与预处理（nnUNet_raw + nnUNetv2_plan_and_preprocess）已抽离到：
- data_process.py

示例：
  python train_nnunet.py \
	--work-dir "./work" \
	--dataset-id 100 \
	--configuration 3d_fullres \
	--plans-name nnUNetResEncUNetMPlans \
	--fold all \
	--epochs 250 \
	--num-gpus 1
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Optional, Sequence, Union


def run(cmd: str, *, timeout: Optional[int] = None) -> None:
	print(f"Running: {cmd}")
	if timeout is not None:
		print(f"Timeout: {timeout}s ({timeout/3600:.1f}h)")
	completed = subprocess.run(cmd, shell=True, timeout=timeout)
	if completed.returncode != 0:
		raise RuntimeError(f"Command failed (exit={completed.returncode}): {cmd}")


def set_nnunet_env(*, nnunet_raw: Path, nnunet_preprocessed: Path, nnunet_results: Path, compile_flag: bool) -> None:
	nnunet_raw.mkdir(parents=True, exist_ok=True)
	nnunet_preprocessed.mkdir(parents=True, exist_ok=True)
	nnunet_results.mkdir(parents=True, exist_ok=True)
	os.environ["nnUNet_raw"] = str(nnunet_raw)
	os.environ["nnUNet_preprocessed"] = str(nnunet_preprocessed)
	os.environ["nnUNet_results"] = str(nnunet_results)
	os.environ["nnUNet_compile"] = "true" if compile_flag else "false"


def trainer_name_for_epochs(epochs: Optional[int]) -> str:
	if epochs is None or int(epochs) == 1000:
		return "nnUNetTrainer"
	if int(epochs) == 1:
		return "nnUNetTrainer_1epoch"
	return f"nnUNetTrainer_{int(epochs)}epochs"


def _find_dataset_preprocessed_dir(nnunet_preprocessed: Path, dataset_id: int) -> Optional[Path]:
	prefix = f"Dataset{dataset_id:03d}_"
	if not nnunet_preprocessed.exists():
		return None
	for p in nnunet_preprocessed.iterdir():
		if p.is_dir() and p.name.startswith(prefix):
			return p
	return None


def get_plans_batch_size(
	*,
	nnunet_preprocessed: Path,
	dataset_id: int,
	plans_name: str,
	configuration: str,
) -> Optional[int]:
	"""读取 nnUNet plans 中指定 configuration 的 batch_size。

	返回 None 表示无法读取（例如未预处理/路径不对/文件不存在）。
	"""
	ds_dir = _find_dataset_preprocessed_dir(nnunet_preprocessed, dataset_id)
	if ds_dir is None:
		return None
	plans_file = ds_dir / f"{plans_name}.json"
	if not plans_file.exists():
		return None
	try:
		with plans_file.open("r", encoding="utf-8") as f:
			plans = json.load(f)
		return int(plans["configurations"][configuration]["batch_size"])
	except Exception:
		return None


def _parse_devices(devices: Optional[str]) -> list[str]:
	if devices is None:
		return []
	items = [x.strip() for x in devices.split(",")]
	items = [x for x in items if x]
	for x in items:
		if not x.isdigit():
			raise ValueError(f"Invalid device id in --devices: {x!r} (expected comma-separated integers)")
	return items


def _default_device_pool_from_env() -> list[str]:
	# If CUDA_VISIBLE_DEVICES is set, respect it as the pool.
	cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
	if not cvd:
		return []
	items = [x.strip() for x in cvd.split(",")]
	return [x for x in items if x]


def _chunk_list(items: list[str], chunk_size: int) -> list[list[str]]:
	if chunk_size <= 0:
		raise ValueError("chunk_size must be > 0")
	return [items[i : i + chunk_size] for i in range(0, len(items), chunk_size)]


def _parse_folds(fold_arg: str) -> list[Union[int, str]]:
	# nnUNet 默认 5-fold CV；nnUNetv2_train 也支持 fold=all。
	# 为了并行启动，我们把 all 展开成 0..4。
	fold_arg = str(fold_arg).strip()
	if fold_arg.lower() == "all":
		return [0, 1, 2, 3, 4]
	# 允许 "0" / "1" ...
	if fold_arg.isdigit():
		return [int(fold_arg)]
	# 允许 "0,1,2" 形式
	if "," in fold_arg:
		out: list[Union[int, str]] = []
		for part in fold_arg.split(","):
			part = part.strip()
			if not part:
				continue
			if not part.isdigit():
				raise ValueError(f"Invalid fold in --fold: {part!r}")
			out.append(int(part))
		return out
	# 兜底：交给 nnUNet 自己处理（例如某些特殊 fold 标记）
	return [fold_arg]


def nnunet_train(
	*,
	dataset_id: int,
	configuration: str,
	fold: Union[int, str],
	plans_name: str,
	epochs: Optional[int],
	num_gpus: int,
	continue_training: bool,
	pretrained_weights: Optional[Path],
	only_val: bool,
	disable_checkpointing: bool,
	npz: bool,
	timeout: Optional[int],
) -> None:
	trainer = trainer_name_for_epochs(epochs)
	cmd = f"nnUNetv2_train {dataset_id:03d} {configuration} {fold} -p {plans_name} -tr {trainer}"
	if pretrained_weights is not None:
		cmd += f" -pretrained_weights \"{pretrained_weights}\""
	if continue_training:
		# nnUNetv2_train 的继续训练参数是 -c
		cmd += " -c"
	if only_val:
		cmd += " --val"
	if disable_checkpointing:
		cmd += " --disable_checkpointing"
	if npz:
		cmd += " --npz"
	if num_gpus and num_gpus > 1:
		cmd += f" -num_gpus {int(num_gpus)}"
	run(cmd, timeout=timeout)


def nnunet_train_parallel_folds(
	*,
	dataset_id: int,
	configuration: str,
	folds: list[Union[int, str]],
	plans_name: str,
	epochs: Optional[int],
	gpus_per_job: int,
	device_pool: list[str],
	continue_training: bool,
	pretrained_weights: Optional[Path],
	only_val: bool,
	disable_checkpointing: bool,
	npz: bool,
	timeout: Optional[int],
) -> None:
	"""并行跑多个 fold，每个 fold 占用 gpus_per_job 张 GPU。

	通过为每个子进程设置 CUDA_VISIBLE_DEVICES 来做 GPU 切分。
	"""
	if gpus_per_job <= 0:
		raise ValueError("--gpus-per-job must be >= 1")
	if len(device_pool) < gpus_per_job:
		raise ValueError(f"Not enough GPUs in pool: need {gpus_per_job}, got {len(device_pool)}")
	chunks = _chunk_list(device_pool, gpus_per_job)
	if not chunks:
		raise ValueError("No GPUs available in device pool")

	# wave execution: at most len(chunks) concurrent jobs
	idx = 0
	while idx < len(folds):
		procs: list[tuple[subprocess.Popen[object], str]] = []
		for slot in range(len(chunks)):
			if idx >= len(folds):
				break
			fold = folds[idx]
			gpu_ids = chunks[slot]
			trainer = trainer_name_for_epochs(epochs)
			cmd = f"nnUNetv2_train {dataset_id:03d} {configuration} {fold} -p {plans_name} -tr {trainer}"
			if pretrained_weights is not None:
				cmd += f" -pretrained_weights \"{pretrained_weights}\""
			if continue_training:
				cmd += " -c"
			if only_val:
				cmd += " --val"
			if disable_checkpointing:
				cmd += " --disable_checkpointing"
			if npz:
				cmd += " --npz"
			if gpus_per_job > 1:
				cmd += f" -num_gpus {int(gpus_per_job)}"

			env = os.environ.copy()
			env["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_ids)
			print(f"[parallel] fold={fold} GPUs={env['CUDA_VISIBLE_DEVICES']} cmd={cmd}")
			p = subprocess.Popen(cmd, shell=True, env=env)
			procs.append((p, cmd))
			idx += 1

		# Wait this wave
		for p, cmd in procs:
			try:
				ret = p.wait(timeout=timeout)
			except subprocess.TimeoutExpired:
				p.kill()
				raise RuntimeError(f"Timeout while running: {cmd}")
			if ret != 0:
				raise RuntimeError(f"Command failed (exit={ret}): {cmd}")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Vesuvius surface nnUNetv2 training script")
	parser.add_argument("--work-dir", type=Path, default=Path("./work"), help="工作目录（默认 ./work）")
	parser.add_argument("--nnunet-raw", type=Path, default=None, help="覆盖 nnUNet_raw 路径")
	parser.add_argument("--nnunet-preprocessed", type=Path, default=None, help="覆盖 nnUNet_preprocessed 路径")
	parser.add_argument("--nnunet-results", type=Path, default=None, help="覆盖 nnUNet_results 路径")

	parser.add_argument("--dataset-id", type=int, default=100)
	parser.add_argument("--configuration", type=str, default="3d_fullres")
	parser.add_argument("--plans-name", type=str, default="nnUNetResEncUNetMPlans")
	parser.add_argument("--fold", type=str, default="all")

	parser.add_argument("--epochs", type=int, default=None, help="使用 nnUNetTrainer_{epochs}epochs；None=默认 1000")
	parser.add_argument("--num-gpus", type=int, default=1)
	parser.add_argument(
		"--parallel-folds",
		type=int,
		default=0,
		choices=[0, 1],
		help=(
			"是否并行跑多个 fold（用多进程占满多张卡）。"
			"当 plans 的 batch_size 很小（例如=2）时，用这个比硬开 8 卡 DDP 更合理。"
		),
	)
	parser.add_argument(
		"--gpus-per-job",
		type=int,
		default=None,
		help="并行 fold 时，每个 fold 占用几张 GPU（默认自动取 min(batch_size, 2) 且至少 1）",
	)
	parser.add_argument(
		"--devices",
		type=str,
		default=None,
		help="并行 fold 时的 GPU 池，例如 '0,1,2,3,4,5,6,7'；若不填则尝试读取 CUDA_VISIBLE_DEVICES",
	)

	parser.add_argument("--continue-training", type=int, default=0, choices=[0, 1])
	parser.add_argument("--pretrained-weights", type=Path, default=None)
	parser.add_argument("--only-val", type=int, default=0, choices=[0, 1])
	parser.add_argument("--disable-checkpointing", type=int, default=0, choices=[0, 1])
	parser.add_argument("--npz", type=int, default=0, choices=[0, 1])

	parser.add_argument("--timeout", type=int, default=None)
	parser.add_argument("--compile", type=int, default=0, choices=[0, 1], help="是否设置 nnUNet_compile=true")
	return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
	args = parse_args(argv)

	work_dir: Path = args.work_dir
	nnunet_raw = args.nnunet_raw or (work_dir / "nnUNet_raw")
	nnunet_preprocessed = args.nnunet_preprocessed or (work_dir / "nnUNet_preprocessed")
	nnunet_results = args.nnunet_results or (work_dir / "nnUNet_results")

	set_nnunet_env(
		nnunet_raw=nnunet_raw,
		nnunet_preprocessed=nnunet_preprocessed,
		nnunet_results=nnunet_results,
		compile_flag=bool(args.compile),
	)

	print("=" * 60)
	print("nnUNet training")
	print("=" * 60)
	print("dataset_id:", args.dataset_id)
	print("configuration:", args.configuration)
	print("plans_name:", args.plans_name)
	print("fold:", args.fold)
	print("epochs:", args.epochs if args.epochs is not None else 1000)
	print("num_gpus:", args.num_gpus)
	print("nnUNet_raw:", nnunet_raw)
	print("nnUNet_preprocessed:", nnunet_preprocessed)
	print("nnUNet_results:", nnunet_results)
	print("=" * 60)

	plans_batch_size = get_plans_batch_size(
		nnunet_preprocessed=nnunet_preprocessed,
		dataset_id=int(args.dataset_id),
		plans_name=str(args.plans_name),
		configuration=str(args.configuration),
	)
	if plans_batch_size is not None:
		print("plans batch_size:", plans_batch_size)
		if int(args.parallel_folds) == 0 and int(args.num_gpus) > int(plans_batch_size):
			raise SystemExit(
				"你请求的 --num-gpus 超过了该 configuration 在 plans 里的 global batch_size，"
				"nnUNet DDP 会直接 assert 失败。\n"
				f"- configuration={args.configuration} batch_size={plans_batch_size}\n"
				f"- 你传入的 num_gpus={args.num_gpus}\n\n"
				"可选解决方案：\n"
				f"1) 把 --num-gpus 降到 <= {plans_batch_size}（通常就是 1 或 2）\n"
				"2) 用 --parallel-folds 1 并行跑多个 fold（例如 8 卡可以 4 个 fold×每个 2 卡）\n"
			)

	if int(args.parallel_folds) == 1:
		folds = _parse_folds(str(args.fold))
		device_pool = _parse_devices(args.devices) if args.devices else _default_device_pool_from_env()
		if not device_pool:
			# fallback: assume 0..num_gpus-1 is the pool
			device_pool = [str(i) for i in range(int(args.num_gpus))]

		allowed = plans_batch_size if plans_batch_size is not None else None
		default_gpj = 1
		if allowed is not None:
			default_gpj = max(1, min(int(allowed), 2))
		gpus_per_job = int(args.gpus_per_job) if args.gpus_per_job is not None else default_gpj
		if allowed is not None and gpus_per_job > int(allowed):
			raise SystemExit(
				f"--gpus-per-job={gpus_per_job} 超过 plans batch_size={allowed}，会触发 DDP assert。"
			)

		print("parallel_folds:", True)
		print("folds:", folds)
		print("device_pool:", ",".join(device_pool))
		print("gpus_per_job:", gpus_per_job)

		nnunet_train_parallel_folds(
			dataset_id=int(args.dataset_id),
			configuration=str(args.configuration),
			folds=folds,
			plans_name=str(args.plans_name),
			epochs=args.epochs,
			gpus_per_job=gpus_per_job,
			device_pool=device_pool,
			continue_training=bool(args.continue_training),
			pretrained_weights=args.pretrained_weights,
			only_val=bool(args.only_val),
			disable_checkpointing=bool(args.disable_checkpointing),
			npz=bool(args.npz),
			timeout=args.timeout,
		)
		print("Done.")
		return 0

	nnunet_train(
		dataset_id=args.dataset_id,
		configuration=args.configuration,
		fold=args.fold,
		plans_name=args.plans_name,
		epochs=args.epochs,
		num_gpus=int(args.num_gpus),
		continue_training=bool(args.continue_training),
		pretrained_weights=args.pretrained_weights,
		only_val=bool(args.only_val),
		disable_checkpointing=bool(args.disable_checkpointing),
		npz=bool(args.npz),
		timeout=args.timeout,
	)

	print("Done.")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
