import numpy as np

# NP-M1 批量计算夹角
np.set_printoptions(precision=6, floatmode="fixed")


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
