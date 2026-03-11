# Vesuvius Surface Detection（本地训练/推理脚本）

本仓库把两条常见 baseline 流程脚本化，便于在本地（含 Windows）训练与生成 `submission.zip`：

- **nnUNetv2 路线**：自动规划/预处理/训练/推理（调用 `nnUNetv2_*` CLI）
- **3D-SegM 路线**：MONAI + PyTorch Lightning 的 3D 分割 baseline（SegResNet / SwinUNETR）

## 目录概览

- `train_nnunet.py`：nnUNet 训练/续训（仅训练阶段；数据准备与预处理见 `data_process.py`）
- `infer.py`：nnUNet 推理 + 生成 `submission.zip`（可选后处理）
- `train_3d_segm.py`：3D-SegM 训练（MONAI + Lightning）
- `infer_3d_segm.py`：3D-SegM 推理 + 生成 `submission.zip`（可选后处理）
- `requirements.txt`：Python 依赖列表（见下方安装说明）

## 数据目录结构（约定）

下面把比赛数据根目录记为 `DATA_ROOT`：

```
DATA_ROOT/
  train_images/
    <case_id>.tif
    ...
  train_labels/
    <case_id>.tif
    ...
  test_images/
    <case_id>.tif
    ...
  test.csv
```

- `train_images` 与 `train_labels` 必须**文件名一一对应**（同名 `.tif/.npy/.npz` 也可用于 3D-SegM）。
- `test.csv` 里 `id` 列决定最终需要打包进 zip 的 mask 文件名。

## 环境安装（Windows 友好）

建议 Python 3.10+。

1) 创建虚拟环境（示例）：

```bash
python -m venv venv
# PowerShell
.\venv\Scripts\Activate.ps1
```

2) 安装 PyTorch

由于 Windows 上 CUDA/CPU 组合差异较大，建议按 PyTorch 官方安装向导选择命令：
- https://pytorch.org/get-started/locally/

3) 安装其余依赖：

```bash
pip install -r requirements.txt
```

> 说明：如果你把 `torch` 放在了 requirements 之外（常见做法），请确保第 2 步已完成，否则 3D-SegM 无法运行。

## 方案 A：3D-SegM（MONAI + Lightning）

### 训练

```bash
python train_3d_segm.py \
  --train-images-dir "DATA_ROOT/train_images" \
  --train-labels-dir "DATA_ROOT/train_labels" \
  --output-dir "./work_3d_segm" \
  --arch segresnet \
  --model-input-size 160 160 160 \
  --max-epochs 20 \
  --batch-size 1 \
  --num-workers 2 \
  --gpu-transforms 1
```

- 断点续训（从 `output-dir` 自动找 `val_dice` 最好的 ckpt 继续）：

```bash
python train_3d_segm.py ... --resume-best 1
```

### 推理 + 打包 submission.zip

`infer_3d_segm.py` 需要 `--root-dir` 指向 `DATA_ROOT`（包含 `test_images/` 与 `test.csv`）。

```bash
python infer_3d_segm.py \
  --root-dir "DATA_ROOT" \
  --checkpoint-dir "./work_3d_segm" \
  --work-dir "./work_3d_segm_infer" \
  --postprocess 1
```

输出：
- `work-dir/submission_masks/*.tif`
- `work-dir/submission.zip`

### （推荐）V1：patch 训练 + 原尺寸滑窗推理

如果你希望避免 “resize 到固定立方体再插回去” 带来的拓扑硬伤，可以使用 v1 脚本：

- 训练：[train_3d_segm_v1.py](train_3d_segm_v1.py)（随机采样 patch，默认 192³）
- 推理：[infer_3d_segm_v1.py](infer_3d_segm_v1.py)（原尺寸 `sliding_window_inference`）

#### V1 训练

```bash
python train_3d_segm_v1.py \
  --train-images-dir "DATA_ROOT/train_images" \
  --train-labels-dir "DATA_ROOT/train_labels" \
  --output-dir "./work_3d_segm_v1" \
  --arch segresnet \
  --patch-size 192 192 192 \
  --samples-per-volume 16 \
  --pos-fraction 0.7 \
  --batch-size 1 \
  --max-epochs 20 \
  --ema 1 \
  --ema-decay 0.999
```

多 GPU 训练（Lightning 原生支持）：

- Linux（推荐 DDP）：

```bash
python train_3d_segm_v1.py \
  --train-images-dir "DATA_ROOT/train_images" \
  --train-labels-dir "DATA_ROOT/train_labels" \
  --output-dir "./work_3d_segm_v1" \
  --devices 4 \
  --strategy ddp
```

- Windows（通常用 spawn）：

```bash
python train_3d_segm_v1.py \
  --train-images-dir "DATA_ROOT/train_images" \
  --train-labels-dir "DATA_ROOT/train_labels" \
  --output-dir "./work_3d_segm_v1" \
  --devices 2 \
  --strategy ddp_spawn
```

关键参数：
- `--patch-size 192 192 192`：训练 patch 尺寸（起步推荐 192³）。
- `--samples-per-volume`：每个 volume 每个 epoch 采多少个 patch（越大越“见得多”，但训练越慢）。
- `--pos-fraction`：pos/neg 引导采样中“采到前景中心”的概率（尽量多采到标注区域，避免全 ignore patch）。
- `--ema/--ema-decay`：启用 EMA；**验证与保存 best 会使用 EMA 权重**，监控指标为 `val_fg_dice`（只算 class=1）。
- `--devices`：使用哪些 GPU（`auto` / `1` / `0,1,2,3`）。
- `--strategy`：分布式策略（`auto` / `ddp` / `ddp_spawn`）。
- `--num-nodes`：多机训练节点数（默认 1）。

#### V1 推理 + 打包 submission.zip（原尺寸滑窗）

```bash
python infer_3d_segm_v1.py \
  --root-dir "DATA_ROOT" \
  --checkpoint-dir "./work_3d_segm_v1" \
  --work-dir "./work_3d_segm_v1_infer" \
  --roi-size 192 192 192 \
  --overlap 0.6 \
  --sw-batch-size 1 \
  --tta 1 \
  --postprocess 1
```

关键参数：
- `--roi-size 192 192 192`：滑窗推理窗口大小；也可试 `224 224 224`。
- `--overlap 0.5~0.7`：越大越平滑、越不容易断裂/块状边界，但更慢更吃显存。
- `--tta 1`：轻扰动 TTA（不翻转），做两次推理取均值（可用 `--tta-noise-std/--tta-scale-jitter` 微调）。

## 方案 B：nnUNetv2

### Step 1/2：数据准备 + 预处理

先把比赛数据转换为 nnUNet 目录结构，并运行 `nnUNetv2_plan_and_preprocess`：

```bash
python data_process.py \
  --input-dir "DATA_ROOT" \
  --work-dir "./work" \
  --dataset-id 100 \
  --configuration 3d_fullres \
  --planner nnUNetPlannerResEncM
```

- 只做“准备数据”（不跑 preprocess）：`--do-preprocess 0`
- 只做“preprocess”（不重新准备数据）：`--do-prepare 0`
- Windows 下如果 symlink 权限有问题：加 `--no-symlinks`

### Step 2/2：训练

```bash
python train_nnunet.py \
  --work-dir "./work" \
  --dataset-id 100 \
  --configuration 3d_fullres \
  --plans-name nnUNetResEncUNetMPlans \
  --fold all \
  --epochs 250 \
  --num-gpus 1
```

#### 多 GPU 注意事项（很重要）

nnUNet 的 DDP 多卡训练有一个硬性约束：**global batch_size 必须 >= GPU 数**（否则会 assert 报错）。

- 例如你贴出来的 `3d_fullres/3d_lowres` 在 plans 里 `batch_size=2`，所以：
  - 单次 DDP **最多只能用 2 张卡**：`--num-gpus 2`
  - 直接 `--num-gpus 8` 一定会失败
- 我已在脚本里加入“提前检查”：当 `--num-gpus` 超过 plans 的 `batch_size` 时，会直接给出更清晰的报错与建议。

#### 用满 8 卡的推荐方式：并行跑多个 fold

当单次 DDP 只能用 1～2 卡时，想把 8 卡吃满，推荐改为“并行启动多个 fold”，每个 fold 占用 1～2 卡：

示例：8 张卡并行跑 4 个 fold（每个 fold 2 卡）：

```bash
python train_nnunet.py \
  --work-dir "./work" \
  --dataset-id 100 \
  --configuration 3d_fullres \
  --plans-name nnUNetResEncUNetMPlans \
  --fold all \
  --epochs 250 \
  --parallel-folds 1 \
  --num-gpus 8 \
  --devices 0,1,2,3,4,5,6,7 \
  --gpus-per-job 2
```

参数说明：
- `--parallel-folds 1`：开启并行 fold 模式（多进程）。
- `--devices`：指定 GPU 池（会为每个子进程设置 `CUDA_VISIBLE_DEVICES`）。
- `--gpus-per-job`：每个 fold 占用几张卡（必须 `<= plans batch_size`；你这个 plans 是 2）。

小技巧：
- 不写 `--devices` 时：会优先尝试读取环境变量 `CUDA_VISIBLE_DEVICES` 作为 GPU 池；如果也没有，会假设 `0..num_gpus-1`。
- `--fold` 除了 `all`，也支持 `0` 或 `0,1,2` 这种形式（方便你只跑部分 fold）。

### 推理 + 打包 submission.zip

```bash
python infer.py \
  --root-dir "DATA_ROOT" \
  --work-dir "./work_infer" \
  --dataset-id 100 \
  --configuration 3d_fullres \
  --plans-name nnUNetResEncUNetMPlans \
  --trainer nnUNetTrainer_250epochs \
  --fold all \
  --checkpoint checkpoint_final.pth \
  --save-probabilities 1 \
  --postprocess 1
```

输出：
- `work-dir/submission_masks/*.tif`
- `work-dir/submission.zip`

## 可选：predictions 风格后处理（两个推理脚本都支持）

开启方式：`--postprocess 1`

默认参数（可按需调整）：
- `--T-low 0.30` / `--T-high 0.80`：3D hysteresis 阈值
- `--z-radius 3` / `--xy-radius 2`：各向异性 closing 结构元
- `--dust-min-size 100`：去除小连通域

依赖：`scipy` + `scikit-image`（已在 requirements 中列出）。

## 常见问题

- **找不到数据/文件名不匹配**：确保 `train_images` 与 `train_labels` 文件名完全一致；并确认 `test.csv` 存在且 `test_images` 下有对应 `.tif`。
- **Windows symlink 失败**：nnUNet 相关脚本可用 `--no-symlinks`，或开启 Windows 开发者模式/管理员权限。
- **显存不够（OOM）**：优先减小 `--model-input-size`（3D-SegM）或减小 batch、关闭 `--gpu-transforms`；nnUNet 则考虑换配置/patch size 或减少并行。
