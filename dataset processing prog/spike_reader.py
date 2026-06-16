# =========================================================
# spike_reader.py
# 可视化.dat脉冲流文件
# 
# 1. 功能
# 可视化查看脉冲流文件
#
# 2. 注意：
# (1) 一次只能看一个文件
# (2) 需要手动改文件名："# DAT 文件路径"下的DAT_PATH
# (3) "# 参数"下的HEIGHT、WIDTH、TIMESTEPS需要与脉冲流生成参数一致
#
# =========================================================
import cv2
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

# =========================================================
# DAT 文件路径
# =========================================================
DAT_PATH = Path(
    # 文件目录
    "luminance_expanded_spike_x4k_test/train/input/" 
    # 文件名
    "lambda1.0_occ000.015_f4881.dat"
)

# =========================================================
# 参数
# =========================================================
HEIGHT = 1000
WIDTH = 1000

# (5帧 × 每帧8) + 1 timestep
TIMESTEPS = 41

# =========================================================
# DAT 读取函数
# =========================================================
def read_dat(dat_path, height, width, timesteps):
    """
    读取 bit-packed DAT 文件
    """
    # 读取字节流
    packed = np.fromfile(dat_path, dtype=np.uint8)

    print(f"packed bytes: {len(packed)}")

    # unpack bits
    bits = np.unpackbits(packed)

    expected_size = timesteps * height * width

    print(f"expected bits: {expected_size}")
    print(f"actual bits  : {len(bits)}")

    # 截断多余 padding
    bits = bits[:expected_size]

    # reshape
    spikes = bits.reshape(
        timesteps,
        height,
        width
    )

    return spikes.astype(np.uint8)

# =========================================================
# 读取 DAT
# =========================================================
print("开始读取 DAT 文件...")
print(DAT_PATH)

spikes = read_dat(
    DAT_PATH,
    HEIGHT,
    WIDTH,
    TIMESTEPS
)

# =========================================================
# 基本信息
# =========================================================
print("\n==============================")
print("读取完成")
print("==============================")

print(f"shape : {spikes.shape}")
print(f"dtype : {spikes.dtype}")

print(f"min   : {spikes.min()}")
print(f"max   : {spikes.max()}")

print(f"总 spike 数量 : {spikes.sum()}")

print(
    f"平均 spike rate : "
    f"{spikes.mean():.6f}"
)

# =========================================================
# 可视化单个 timestep
# =========================================================
VIS_T = 10

frame = spikes[VIS_T]

frame_vis = (frame * 255).astype(np.uint8)

plt.figure(figsize=(6, 6)) 

plt.imshow(frame_vis, cmap="gray") 

plt.title(f"Spike Frame T={VIS_T}") 

plt.axis("on") 

plt.show()

# =========================================================
# Spike 累积图
# =========================================================
accumulate = spikes.sum(axis=0)

accumulate = accumulate.astype(np.float32)

if accumulate.max() > 0:
    accumulate /= accumulate.max()

plt.figure(figsize=(8, 6))

plt.imshow(accumulate, cmap="gray")

plt.title("Spike Accumulation")

plt.axis("on")

plt.show()

# =========================================================
# 保存累积图
# =========================================================
save_img = (accumulate * 255).astype(np.uint8)

save_path = DAT_PATH.with_suffix(".png")

cv2.imwrite(str(save_path), save_img)

print("\n累积图已保存:")
print(save_path)

print("\n===================================")
print("DAT 文件验证完成")
print("===================================")
