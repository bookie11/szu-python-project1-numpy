import numpy as np

# NP-M3 所有点对之间的距离矩阵
np.set_printoptions(precision=6, floatmode="fixed")

A = np.array([[0.20, -0.40],
              [0.80, 0.30],
              [-0.50, 0.70]])
B = np.array([[0.10, 0.20],
              [-0.60, -0.20]])

print("========== NP-M3 所有点对之间的距离矩阵 ==========")
print(f"A =\n{A}")
print(f"A.shape = {A.shape}")
print(f"B =\n{B}")
print(f"B.shape = {B.shape}")

# 用 np.newaxis 扩展维度后做广播相减：
# A[:, np.newaxis, :] 的 shape 是 (3, 1, 2)
# B[np.newaxis, :, :] 的 shape 是 (1, 2, 2)
# 差数组 diff 的 shape 是 (3, 2, 2)，其中 diff[i, j] = A[i] - B[j]。
diff = A[:, np.newaxis, :] - B[np.newaxis, :, :]
print(f"差数组 diff.shape = {diff.shape}（对应 (len(A), len(B), 坐标维数)）")

# 沿最后一维（坐标维 axis=-1）做平方和再开平方，得到距离矩阵 D，shape (3, 2)。
D = np.sqrt(np.sum(diff ** 2, axis=-1))
print(f"距离矩阵 D =\n{D}")
print(f"D.shape = {D.shape}")

# 沿 axis=1 在每一行中找最小距离对应的列下标，即离 A[i] 最近的 B 行索引，shape (3,)。
nearest_idx = np.argmin(D, axis=1)
print(f"A 每行距离最近的 B 行索引 = {nearest_idx}，shape = {nearest_idx.shape}")

print("---------- 边界输入：点集与自身计算距离矩阵 ----------")
self_diff = A[:, np.newaxis, :] - A[np.newaxis, :, :]
self_D = np.sqrt(np.sum(self_diff ** 2, axis=-1))
print(f"A 到自身的距离矩阵 =\n{self_D}")
print(f"最近索引 = {np.argmin(self_D, axis=1)}（对角线为 0，每行最近点都是自己）")
