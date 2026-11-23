import numpy as np

# NP-M4 用矩阵运算完成一次前向计算
np.set_printoptions(precision=6, floatmode="fixed")

X = np.array([[0.80, -0.60, 0.70],
              [-0.40, -0.50, 0.90]])
W1 = np.array([[-0.10, 0.20, -0.80],
               [0.30, -0.50, 0.10]])
b1 = np.array([-0.15, 0.05])
W2 = np.array([[0.20, -0.60],
               [-0.80, 0.52]])
b2 = np.array([0.10, -0.20])

print("========== NP-M4 用矩阵运算完成一次前向计算 ==========")
print(f"X.shape = {X.shape}，W1.shape = {W1.shape}，b1.shape = {b1.shape}")
print(f"W2.shape = {W2.shape}，b2.shape = {b2.shape}")

# 第一层线性运算：X (2, 3) @ W1.T (3, 2) 得到 (2, 2)，再广播加上 b1 (2,)，得到 Z (2, 2)。
Z = X @ W1.T + b1
# 逐元素 ReLU：负数置 0，正数保持不变，shape 仍为 (2, 2)。
H = np.maximum(Z, 0.0)
# 第二层线性运算：H (2, 2) @ W2.T (2, 2) + b2 (2,)，得到输出 Y (2, 2)。
Y = H @ W2.T + b2

print(f"Z =\n{Z}")
print(f"Z.shape = {Z.shape}")
print(f"H =\n{H}")
print(f"H.shape = {H.shape}")
print(f"Y =\n{Y}")
print(f"Y.shape = {Y.shape}")

print("---------- 边界输入：全零单样本输入 ----------")
X_zero = np.array([[0.00, 0.00, 0.00]])
Z_zero = X_zero @ W1.T + b1
H_zero = np.maximum(Z_zero, 0.0)
Y_zero = H_zero @ W2.T + b2
print(f"X_zero = {X_zero}，shape = {X_zero.shape}")
print(f"Z_zero = {Z_zero}（等于 b1），H_zero = {H_zero}，Y_zero = {Y_zero}，shape = {Y_zero.shape}")
