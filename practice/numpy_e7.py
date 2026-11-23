import numpy as np

# NP-E7 一次广播，处理所有行
np.set_printoptions(precision=6, floatmode="fixed")

X = np.array([[0.80, -0.60, 0.70],
              [-0.40, 0.50, 0.20]])
scale = np.array([1.50, 0.50, -1.00])
bias = np.array([-0.10, 0.20, 0.30])

print("========== NP-E7 一次广播，处理所有行 ==========")
print(f"X =\n{X}")
print(f"X.shape = {X.shape}")
print(f"scale = {scale}，shape = {scale.shape}")
print(f"bias = {bias}，shape = {bias.shape}")

# 广播对齐规则：从最右边（最后一维）开始对齐。
# X 的 shape 是 (2, 3)，scale 和 bias 的 shape 是 (3,)，
# NumPy 会先在左侧补 1 视为 (1, 3)，再沿第 0 轴扩展为 (2, 3)，
# 因此每一行的 3 个特征都会分别乘以对应的 scale[j] 再加上 bias[j]。
Y = X * scale + bias
print(f"Y = X * scale + bias =\n{Y}")
print(f"Y.shape = {Y.shape}")

print("广播对齐说明：")
print("  - scale (3,) 和 bias (3,) 先右对齐为 (1, 3)，再沿行方向广播为 (2, 3)")
print("  - 第 0 行与第 1 行各自的第 j 列特征，都使用相同的 scale[j] 和 bias[j]")

print("---------- 边界输入：单行二维数组 ----------")
X_single = np.array([[1.00, 2.00, 3.00]])
Y_single = X_single * scale + bias
print(f"X_single = {X_single}，shape = {X_single.shape}")
print(f"Y_single = {Y_single}，shape = {Y_single.shape}")
