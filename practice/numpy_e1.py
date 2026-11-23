import numpy as np

# NP-E1 数组、形状与切片
# 讲评：只改打印精度，不改变真实计算结果。
np.set_printoptions(precision=6, floatmode="fixed")

A = np.array([[0.20, -0.50, 1.10, 0.40],
              [0.80, 0.30, -0.70, 0.60],
              [-0.40, 0.90, 0.10, -0.20]])

print("========== NP-E1 数组、形状与切片 ==========")
# 输入 shape (3, 4)：3 行 4 列。ndim 是维数，size 是元素总个数 3*4=12。
print(f"输入 A =\n{A}")
print(f"ndim = {A.ndim}")
print(f"shape = {A.shape}")
print(f"size = {A.size}")

# A[1] 取出第 2 行，少掉“行”这一维，shape 从 (3, 4) 变成 (4,)。
row = A[1]
# A[:, -2:]：冒号表示所有行，-2: 表示最后 2 列，shape 是 (3, 2)。
cols = A[:, -2:]
print(f"第 2 行 = {row}，shape = {row.shape}")
print(f"最后 2 列 =\n{cols}")
print(f"最后 2 列 shape = {cols.shape}")

# copy() 复制出一块新数据。如果写成 B = A，两个名字指向同一块数据，改 B 会改到 A。
B = A.copy()
# B < 0 得到同样 shape 的 True/False 数组，再赋值只会改 True 的位置。
B[B < 0] = 0
print(f"B =\n{B}")
print(f"B.shape = {B.shape}")
print(f"A 保持不变 =\n{A}")

print("---------- 边界输入：不复制就修改 ----------")
edge = np.array([[-1.0, 2.0]])
print(f"边界输入 edge = {edge}，shape = {edge.shape}")
alias = edge
alias[alias < 0] = 0
print(f"改 alias 之后，edge 也变成 {edge}")
