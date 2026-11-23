import numpy as np

# NP-E6 沿不同 axis 汇总
np.set_printoptions(precision=6, floatmode="fixed")

A = np.array([[0.20, -0.50, 1.10],
              [0.80, 0.40, -0.30]])

print("========== NP-E6 沿不同 axis 汇总 ==========")
# 输入 shape (2, 3)：2 行 3 列。
print(f"A =\n{A}")
print(f"A.shape = {A.shape}")

# axis=0 沿第 0 轴（行方向）压缩，把同一列的所有行相加。
# 结果第 j 个元素对应输入第 j 列 A[:, j]，shape 从 (2, 3) 变成 (3,)。
col_sum = A.sum(axis=0)
print(f"按列求和 A.sum(axis=0) = {col_sum}，shape = {col_sum.shape}")

# axis=1 沿第 1 轴（列方向）压缩，把同一行的所有列求平均。
# 结果第 i 个元素对应输入第 i 行 A[i, :]，shape 从 (2, 3) 变成 (2,)。
row_mean = A.mean(axis=1)
print(f"按行求均值 A.mean(axis=1) = {row_mean}，shape = {row_mean.shape}")

# 不指定 axis 时对整个数组所有元素汇总，结果是标量，shape ()。
max_val = A.max()
print(f"整体最大值 A.max() = {max_val:.6f}，shape = {np.shape(max_val)}")

print("对应位置说明：")
print("  - A.sum(axis=0) 的第 j 个值 = A[0, j] + A[1, j]（每一列所有行求和）")
print("  - A.mean(axis=1) 的第 i 个值 = mean(A[i, 0], A[i, 1], A[i, 2])（每一行所有列求均值）")

print("---------- 边界输入：只有 1 行的二维数组 ----------")
edge = np.array([[1.0, -2.0, 3.0]])
print(f"edge = {edge}，shape = {edge.shape}")
print(f"edge.sum(axis=0) = {edge.sum(axis=0)}，shape = {edge.sum(axis=0).shape}")
print(f"edge.mean(axis=1) = {edge.mean(axis=1)}，shape = {edge.mean(axis=1).shape}")
print(f"edge.max() = {edge.max():.6f}，shape = {np.shape(edge.max())}")
