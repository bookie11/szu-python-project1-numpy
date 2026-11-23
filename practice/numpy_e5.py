import numpy as np

# NP-E5 把向量变为单位长度
np.set_printoptions(precision=6, floatmode="fixed")


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
