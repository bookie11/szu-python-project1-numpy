import numpy as np

# NP-E8 把两张图像组成一个 batch
np.set_printoptions(precision=6, floatmode="fixed")

a = np.array([[[0.20, -0.40], [0.70, 0.10]],
              [[0.30, 0.80], [-0.20, 0.50]],
              [[-0.10, 0.60], [0.40, -0.30]]])
b = a + 0.10

print("========== NP-E8 把两张图像组成一个 batch ==========")
# 单张图像按 (C, H, W) = (3, 2, 2) 组织。
print(f"a.shape = {a.shape}，b.shape = {b.shape}")

# np.stack(..., axis=0) 新增第 0 轴作为 batch 维，shape 从两个 (3, 2, 2) 变成 (2, 3, 2, 2)，即 (B, C, H, W)。
batch = np.stack([a, b], axis=0)
print(f"batch.shape (B, C, H, W) = {batch.shape}")

# 提取所有图像的蓝色通道（C=2），保留 B、H、W 三维，shape 为 (2, 2, 2)。
blue_channel = batch[:, 2, :, :]
print(f"两张图像的蓝色通道 (C=2) =\n{blue_channel}")
print(f"蓝色通道 shape = {blue_channel.shape}")

# 将轴顺序从 (B, C, H, W) 转为 (B, H, W, C)，即把原来的轴 (0, 1, 2, 3) 重排为 (0, 2, 3, 1)。
batch_bhwc = np.transpose(batch, (0, 2, 3, 1))
print(f"转为 (B, H, W, C) 后的 shape = {batch_bhwc.shape}")

print("---------- 边界输入：单张图像组成的 batch ----------")
single_batch = np.stack([a], axis=0)
single_bhwc = np.transpose(single_batch, (0, 2, 3, 1))
print(f"单图 batch.shape (B, C, H, W) = {single_batch.shape}")
print(f"单图转为 (B, H, W, C) 后的 shape = {single_bhwc.shape}")
