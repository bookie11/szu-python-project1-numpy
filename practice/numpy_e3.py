import numpy as np

# NP-E3 两个向量的夹角
np.set_printoptions(precision=6, floatmode="fixed")


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
