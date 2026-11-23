import numpy as np

# NP-E4 两个点的距离
np.set_printoptions(precision=6, floatmode="fixed")

p = np.array([-0.50, 1.20, 2.00])
q = np.array([0.80, -0.40, 1.10])

print("========== NP-E4 两个点的距离 ==========")
print(f"p = {p}，shape = {p.shape}")
print(f"q = {q}，shape = {q.shape}")

# 对应坐标相减，仍是 3 个数，shape (3,)。
diff = p - q
print(f"差向量 p - q = {diff}，shape = {diff.shape}")

# 欧氏距离：差的平方和再开平方，3 个数合成 1 个数，shape ()。
euclidean = np.sqrt(np.sum(diff ** 2))
# 曼哈顿距离：差的绝对值相加，同样是标量。
manhattan = np.sum(np.abs(diff))
print(f"欧氏距离 = {euclidean:.6f}，shape = {np.shape(euclidean)}")
print(f"曼哈顿距离 = {manhattan:.6f}，shape = {np.shape(manhattan)}")

print("---------- 边界输入：两个点相同 ----------")
same = np.array([1.0, 2.0, 3.0])
same_diff = same - same
print(f"same = {same}，shape = {same.shape}")
print(f"差向量 = {same_diff}，shape = {same_diff.shape}")
print(f"欧氏距离 = {np.sqrt(np.sum(same_diff ** 2)):.6f}")
print(f"曼哈顿距离 = {np.sum(np.abs(same_diff)):.6f}")
