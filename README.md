# Project 1: OpenCLIP ViT-B/32 图像编码器（NumPy）

深圳大学《Python 程序设计》Project 1 作业代码与实验报告源码。

- 姓名：胡书奇（2025040061）
- 指导教师：樊超

> 注：模型权重 `weights/image_encoder.npz` 和参考数据 `assets/reference_outputs.npz` 体积较大，按作业要求不上传至仓库。

## 目录说明

```text
.
├── image_encoder.py          # 图像编码器实现（TODO-01 ~ TODO-06）
├── README.md                 # 运行说明
├── practice/                 # 课前 12 道 NumPy 练习（NP-E1 ~ NP-E8, NP-M1 ~ NP-M4）
│   ├── numpy_homework.py     # 12 题合并脚本
│   └── numpy_e1.py ~ m4.py   # 单题脚本
└── report/                   # 实验报告 Overleaf 源码（XeLaTeX）
    ├── main.tex
    └── figs/vit_overview.png
```

## 实现模块

在 `image_encoder.py` 中用 NumPy 完成了 `TODO-01` 至 `TODO-06`：

1. `ImageEncoder.patchify`（`TODO-01`）：将 `[B, 3, 224, 224]` 切分重排为 `[B, 49, 3072]`。
2. `MultiHeadAttention.split_heads`（`TODO-02`）：将 `[B, N, 768]` 拆分为 `[B, 12, N, 64]`。
3. `MultiHeadAttention.__call__`（`TODO-03`）：计算缩放点积注意力得分、加权求和、多头合并及输出线性映射。
4. `TransformerBlock.__call__`（`TODO-04`）：完成两次残差相加与两层 GELU MLP 计算。
5. `ImageEncoder.embed`（`TODO-05`）：完成无偏置 Patch 投影、拼接 `CLS` 行并加上位置编码，输出 `[B, 50, 768]`。
6. `ImageEncoder.__call__`（`TODO-06`）：取出 `CLS` 向量，经过 `post_norm` 与 `projection` 映射后归一化为 `[B, 512]`。

## 运行方法

安装依赖（Python 3.9+）：

```bash
python3 -m pip install numpy Pillow imageio
```

将 `image_encoder.py` 放到课程下发的 `project1-numpy-student/` 目录中（与 `helpers.py`、`check.py`、`run.py`、`weights/`、`assets/` 同级），运行检查与推理：

```bash
# 全阶段数值校验
python3 check.py --stage all

# 生成 10 张默认测试图的特征向量
python3 run.py --output output_tokens.npz

# 指定图片推理
python3 run.py --images assets/images/image_01.jpg assets/images/image_02.jpg --output my_tokens.npz
```

运行课前 NumPy 练习：

```bash
python3 practice/numpy_homework.py
```

## `check.py --stage all` 输出结果

```text
assets       PASS: 152 matching visual tensors and ten image/reference pairs
preprocess   PASS shape=(10, 3, 224, 224) max_abs=0.000e+00 mean_abs=0.000e+00 (atol=2e-06, rtol=0.0002)
patches      PASS shape=(2, 49, 3072) max_abs=0.000e+00 mean_abs=0.000e+00 (atol=0.0002, rtol=0.0002)
tokens       PASS shape=(2, 50, 768) max_abs=0.000e+00 mean_abs=0.000e+00 (atol=0.0002, rtol=0.0002)
block0       PASS shape=(2, 50, 768) max_abs=7.540e-06 mean_abs=2.274e-07 (atol=0.0002, rtol=0.0002)
last_tokens  PASS shape=(2, 50, 768) max_abs=4.387e-05 mean_abs=2.406e-06 (atol=0.0002, rtol=0.0002)
output       PASS shape=(10, 512) max_abs=6.780e-07 mean_abs=8.215e-08 (atol=2e-05, rtol=0.0002)
batch        PASS shape=(3, 512) max_abs=4.619e-07 mean_abs=8.133e-08 (atol=2e-05, rtol=0.0002)
Requested numerical checks passed. The reference file is never an encoder input.
```
