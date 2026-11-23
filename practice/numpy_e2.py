import numpy as np

# NP-E2 逐元素乘法、点乘与长度
np.set_printoptions(precision=6, floatmode="fixed")

a = np.array([0.80, -0.60, 0.70])
b = np.array([-0.20, 0.90, 0.40])

print("========== NP-E2 逐元素乘法、点乘与长度 ==========")
print(f"a = {a}，shape = {a.shape}")
print(f"b = {b}，shape = {b.shape}")

# * 是逐个位置相乘，结果仍是 3 个数，shape (3,)。
elementwise = a * b
# @ 是点乘：对应位置相乘后再相加，3 个数合成 1 个数，shape 是 ()。
dot = a @ b
print(f"a * b = {elementwise}，shape = {elementwise.shape}")
print(f"a @ b = {dot:.6f}，shape = {np.shape(dot)}")

# 欧氏长度 = 每个分量先平方，再求和，再开平方。结果是标量，shape ()。
length_a = np.sqrt(np.sum(a ** 2))
length_b = np.sqrt(np.sum(b ** 2))
print(f"a 的长度 = {length_a:.6f}，shape = {np.shape(length_a)}")
print(f"b 的长度 = {length_b:.6f}，shape = {np.shape(length_b)}")

norm_a = np.linalg.norm(a)
norm_b = np.linalg.norm(b)
print(f"np.linalg.norm(a) = {norm_a:.6f}，是否一致：{np.allclose(length_a, norm_a)}")
print(f"np.linalg.norm(b) = {norm_b:.6f}，是否一致：{np.allclose(length_b, norm_b)}")

print("---------- 边界输入：零向量 ----------")
zero = np.array([0.0, 0.0, 0.0])
zero_length = np.sqrt(np.sum(zero ** 2))
print(f"zero = {zero}，shape = {zero.shape}")
print(f"零向量长度 = {zero_length:.6f}，shape = {np.shape(zero_length)}")
print(f"零向量和 a 的点乘 = {a @ zero:.6f}，shape = {np.shape(a @ zero)}")
