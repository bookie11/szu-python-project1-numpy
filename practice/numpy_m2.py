import numpy as np

# NP-M2 用余弦相似度做 Top-k 检索
np.set_printoptions(precision=6, floatmode="fixed")


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
