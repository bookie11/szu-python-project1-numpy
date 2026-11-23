"""NumPy 数值计算练习合集（8 道基础题 NP-E1~NP-E8 + 4 道中等题 NP-M1~NP-M4）。"""
import numpy as np
np.set_printoptions(precision=6, floatmode="fixed")

# NP-E1 数组、形状与切片
# 讲评：只改打印精度，不改变真实计算结果。

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


# NP-E2 逐元素乘法、点乘与长度

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


# NP-E3 两个向量的夹角


def angle_deg(a, b):
    """返回两个同维非零一维向量的夹角，单位为度。"""
    # 输入都是 shape (n,) 的一维数组，输出是一个数，shape ()。
    length_a = np.sqrt(np.sum(a ** 2))
    length_b = np.sqrt(np.sum(b ** 2))
    if length_a == 0 or length_b == 0:
        raise ValueError("零向量没有方向，不能计算夹角")

    # cos(theta) = 点乘 / (两个长度相乘)。点乘是标量，相除后仍是标量。
    cos_theta = (a @ b) / (length_a * length_b)
    # 浮点误差可能让余弦略微超出 [-1, 1]，clip 把它拉回合法范围。
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    return np.degrees(np.arccos(cos_theta))


a = np.array([1.00, -1.00])
b = np.array([1.00, 0.00])

print("========== NP-E3 两个向量的夹角 ==========")
print(f"a = {a}，shape = {a.shape}")
print(f"b = {b}，shape = {b.shape}")

angle = angle_deg(a, b)
print(f"夹角 = {angle:.6f} 度，shape = {np.shape(angle)}")

print("---------- 边界输入：零向量 ----------")
zero = np.array([0.00, 0.00])
print(f"zero = {zero}，shape = {zero.shape}")
try:
    angle_deg(zero, b)
except ValueError as error:
    print(f"捕获到 ValueError：{error}")


# NP-E4 两个点的距离

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


# NP-E5 把向量变为单位长度


def unit_vector(a):
    """返回方向相同、长度为 1 的新数组。"""
    # 输入 shape (n,)，输出 shape 仍是 (n,)，只是长度变成 1。
    length = np.sqrt(np.sum(a ** 2))
    if length == 0:
        raise ValueError("零向量没有方向，不能变成单位向量")
    # 除法会生成新数组，不会改掉原来的 a。
    return a / length


a = np.array([-1.20, 0.40, 0.80])

print("========== NP-E5 把向量变为单位长度 ==========")
print(f"a = {a}，shape = {a.shape}")

unit = unit_vector(a)
print(f"单位向量 = {unit}，shape = {unit.shape}")
print(f"原数组未改变 = {a}")
# allclose 允许 1e-6 级别的误差，用来核对长度是不是 1。
length_is_one = np.allclose(np.linalg.norm(unit), 1.0)
print(f"长度是否为 1：{length_is_one}")

print("---------- 边界输入：零向量 ----------")
zero = np.array([0.0, 0.0, 0.0])
print(f"zero = {zero}，shape = {zero.shape}")
try:
    unit_vector(zero)
except ValueError as error:
    print(f"捕获到 ValueError：{error}")


# NP-E6 沿不同 axis 汇总

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


# NP-E7 一次广播，处理所有行

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


# NP-E8 把两张图像组成一个 batch

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


# NP-M1 批量计算夹角


def angles_to(query, X):
    """计算矩阵 X 中每一行与查询向量 query 之间的夹角（单位：度）。"""
    # 输入 query: shape (D,)，X: shape (N, D)；输出 angles: shape (N,)。
    query_norm = np.linalg.norm(query)
    if query_norm == 0:
        raise ValueError("查询向量 query 不能为零向量")

    # 矩阵乘法 X @ query 一次算出每行与 query 的点乘，shape (N,)。
    dots = X @ query
    # 沿 axis=1 按行求每一行的欧氏长度，shape (N,)。
    x_norms = np.linalg.norm(X, axis=1)

    # 先用 nan 初始化，零向量行自动保留为 np.nan，不发生除零。
    angles = np.full(X.shape[0], np.nan, dtype=float)
    valid = x_norms > 0

    # 仅对非零行计算余弦值，clip 到 [-1, 1] 防止浮点误差越界，再转成角度。
    cos_theta = dots[valid] / (x_norms[valid] * query_norm)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    angles[valid] = np.degrees(np.arccos(cos_theta))
    return angles


query = np.array([0.60, -0.80, 0.00])
X = np.array([[0.30, -0.40, 0.00],
              [-0.60, 0.80, 0.00],
              [0.80, 0.60, 0.00],
              [0.00, 0.00, 0.00]])

print("========== NP-M1 批量计算夹角 ==========")
print(f"query = {query}，shape = {query.shape}")
print(f"X =\n{X}")
print(f"X.shape = {X.shape}")

angles = angles_to(query, X)
print(f"夹角（度）= {angles}，shape = {angles.shape}")

print("---------- 边界输入：零查询与全零矩阵 ----------")
zero_query = np.array([0.00, 0.00, 0.00])
try:
    angles_to(zero_query, X)
except ValueError as error:
    print(f"零查询捕获到 ValueError：{error}")

all_zero_X = np.zeros((2, 3))
print(f"全零矩阵输出 = {angles_to(query, all_zero_X)}，shape = {angles_to(query, all_zero_X).shape}")


# NP-M2 用余弦相似度做 Top-k 检索


def top_k_cosine(query, gallery, k):
    """按与 query 的余弦相似度从高到低返回前 k 个有效行的索引与相似度。"""
    # 输入 query: shape (D,)，gallery: shape (N, D)；
    # 输出 top_indices: shape (k,)，top_scores: shape (k,)。
    query_norm = np.linalg.norm(query)
    if query_norm == 0:
        raise ValueError("查询向量 query 不能为零向量")

    # 找出非零行索引，零向量行不参与相似度计算与排序。
    row_norms = np.linalg.norm(gallery, axis=1)
    valid_indices = np.flatnonzero(row_norms > 0)

    if isinstance(k, bool) or not isinstance(k, (int, np.integer)) or k <= 0 or k > valid_indices.size:
        raise ValueError(f"k 必须为正整数且不超过有效行数 {valid_indices.size}")

    # 仅对有效行做点乘并除以两边长度，得到余弦相似度，shape (M,)。
    valid_dots = gallery[valid_indices] @ query
    valid_scores = valid_dots / (row_norms[valid_indices] * query_norm)

    # 对 -valid_scores 做稳定升序排序，等价于按相似度降序且同分保持原行顺序。
    order = np.argsort(-valid_scores, kind="stable")
    top_order = order[:k]

    top_indices = valid_indices[top_order]
    top_scores = valid_scores[top_order]
    return top_indices, top_scores


query = np.array([0.60, -0.80, 0.00])
gallery = np.array([[0.60, -0.80, 0.00],
                    [0.30, -0.40, 0.00],
                    [0.80, 0.60, 0.00],
                    [-0.60, 0.80, 0.00],
                    [0.00, 0.00, 0.00]])
ids = np.array(["img_a", "img_b", "img_c", "img_d", "img_zero"])
k = 3

print("========== NP-M2 用余弦相似度做 Top-k 检索 ==========")
print(f"query = {query}，shape = {query.shape}")
print(f"gallery.shape = {gallery.shape}，k = {k}")

top_indices, top_scores = top_k_cosine(query, gallery, k)
selected_ids = ids[top_indices]
print(f"Top-{k} 行索引 = {top_indices}，shape = {top_indices.shape}")
print(f"Top-{k} 相似度 = {top_scores}，shape = {top_scores.shape}")
print(f"选中 ID = {selected_ids}，shape = {selected_ids.shape}")

print("---------- 边界输入：非法 k 与零查询 ----------")
for bad_k in (0, 5):
    try:
        top_k_cosine(query, gallery, bad_k)
    except ValueError as error:
        print(f"k={bad_k} 捕获到 ValueError：{error}")

try:
    top_k_cosine(np.zeros(3), gallery, 1)
except ValueError as error:
    print(f"零查询捕获到 ValueError：{error}")


# NP-M3 所有点对之间的距离矩阵

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


# NP-M4 用矩阵运算完成一次前向计算

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

