# =========================================================
# spike_generator_multi_thread.py
# 用数据集图片生成.dat脉冲流文件
# 
# 1. 功能
# 使用官方 Spike库 的 img_to_spike函数 将.png图片组转换为.dat脉冲流文件
# 
# 2. 解释器要求：
# (1) 安装北大官方的Spike库
#   安装过程（在anaconda环境命令行中依次运行）：
#       git clone https://git.openi.org.cn/Cordium/SpikeCV.git （一台pc安装一次SpikeCV即可，剩下的需要每个环境执行一次）
#       cd SpikeCV （安装好的SpikeCV是个文件夹，进入该目录执行下一条命令）
#       pip install -e . （文件夹里预存了配置信息，运行这条命令自动安装配置，但是numpy版本不对需要手动改）
# （安装后实际调用的时候，程序报错区可能还会显示“无法解析导入Spike”，无视掉，正常跑就行）
# (2) pyhton版本为3.10.x
# (3) 安装numpy v1.26.x
# 
# 3. 时间结构：
# (1) 全0空白timestep
# (2) 一组5张图，每张图生成8个timestep
# (3) 一个.dat文件共含41timestep
# 
# 4. 路径要求
# (1) 输入目录：
#     "./luminance_expanded_spike_x4k/train/gt",
#     "./luminance_expanded_spike_x4k/test/gt"
# (2) 输出目录：
#     "./luminance_expanded_spike_x4k/train/input"
#     "./luminance_expanded_spike_x4k/test/input"
# 
# 5. 文件命名格式
# (1) 输入文件：
#     "./luminance_expanded_spike_x4k/train/gt"下："lambda[光度倍率]_occ[序号1].[序号2]_f[序号3]_key_id[id号].png"
#     "./luminance_expanded_spike_x4k/test/gt"下："lambda[光度倍率]_TEST[序号1]_[序号2]_f[序号3]_key_id[id号].png"
# (2) 输出文件：
#     "./luminance_expanded_spike_x4k/train/input"下："lambda[光度倍率]_occ[序号1].[序号2]_f[序号3].dat"
#     "./luminance_expanded_spike_x4k/test/input"下："lambda[光度倍率]_TEST[序号1]_[序号2]_f[序号3].dat"
# 
# 6. 注意：
# 本程序按需求对SpikeCV\SpikeCV\spkData\convert_img.py中的img_to_spike函数做了修改，以适配多线程加速。
# (目录例：本机地址为C:\Users\Lenovo\SpikeCV\SpikeCV\spkData\convert_img.py)
# 若后续需要使用原本的功能，请记得改回文件。
# 
# 7. 采用多线程加速
# 
# =========================================================
import re
import cv2
import numpy as np

from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

# SpikeCV 官方 simulator
from SpikeCV.spkData.convert_img import img_to_spike

# 多线程
from concurrent.futures import ProcessPoolExecutor


# =========================================================
# 参数
# =========================================================
ROOT_DIR = Path("luminance_expanded_spike_x4k")

MODES = ["test", "train"]

VALID_IDS = [7, 14, 21, 28, 35]

# SpikeCV simulator 参数
GAIN_AMP = 0.5
V_TH = 1.0
N_TIMESTEP = 8

# 多进程 worker 数
# 适配你当前 16GB RAM
NUM_WORKERS = 31


# =========================================================
# 文件名匹配
# =========================================================
TRAIN_PATTERN = re.compile(
    r"lambda(?P<lambda>.+?)_occ(?P<occ1>\d+)\.(?P<occ2>\d+)_f(?P<f>\d+)_key_id(?P<id>\d+)\.png"
)

TEST_PATTERN = re.compile(
    r"lambda(?P<lambda>.+?)_TEST(?P<t1>\d+)_(?P<t2>\d+)_f(?P<f>\d+)_key_id(?P<id>\d+)\.png"
)


# =========================================================
# 读取灰度图
# =========================================================
def load_gray_image(img_path):
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise ValueError(f"无法读取图像: {img_path}")

    img = img.astype(np.float32) / 255.0

    return img


# =========================================================
# 使用 SpikeCV 官方 simulator
# =========================================================
def images_to_spikes(image_list):
    """
    image_list:
        list[np.ndarray]
        连续5帧灰度图

    return:
        spikes:
            shape = [T,H,W]
    
    时间结构：
        t=0: 全0空白帧 
        t=1~8: id7 
        t=9~16: id14 
        ...
    """
    spike_all = []

    # ===================================================== 
    # 先加入一个全0 timestep 
    # ===================================================== 
    h, w = image_list[0].shape 

    zero_frame = np.zeros(
        (1, h, w), 
        dtype=np.uint8 
    ) 

    spike_all.append(zero_frame)

    # ===================================================== 
    # 5张图分别生成8 timestep 
    # =====================================================
    for img in image_list:
        sim_spike = img_to_spike(
            img,
            gain_amp=GAIN_AMP,
            v_th=V_TH,
            n_timestep=N_TIMESTEP
        )

        spike_all.append(sim_spike)

    # =====================================================
    # 时间维拼接
    # =====================================================
    spikes = np.concatenate(spike_all, axis=0)

    return spikes


# =========================================================
# 保存 DAT
# =========================================================
def save_dat(spikes, save_path):
    """
    spikes:
        [T,H,W]
        uint8
    """
    # flatten
    flat_spikes = spikes.reshape(-1)

    # packbits
    packed = np.packbits(flat_spikes)

    # 写入 dat
    packed.tofile(str(save_path))


# =========================================================
# train 文件解析
# =========================================================
def parse_train_name(filename):
    match = TRAIN_PATTERN.match(filename)

    if match is None:
        return None

    d = match.groupdict()

    return {
        "lambda": d["lambda"],
        "occ1": d["occ1"],
        "occ2": d["occ2"],
        "f": d["f"],
        "id": int(d["id"]),
    }


# =========================================================
# test 文件解析
# =========================================================
def parse_test_name(filename):
    match = TEST_PATTERN.match(filename)

    if match is None:
        return None

    d = match.groupdict()

    return {
        "lambda": d["lambda"],
        "t1": d["t1"],
        "t2": d["t2"],
        "f": d["f"],
        "id": int(d["id"]),
    }


# =========================================================
# train group key
# =========================================================
def train_group_key(info):
    return (
        info["lambda"],
        info["occ1"],
        info["occ2"],
        info["f"],
    )


# =========================================================
# test group key
# =========================================================
def test_group_key(info):
    return (
        info["lambda"],
        info["t1"],
        info["t2"],
        info["f"],
    )


# =========================================================
# 输出文件名
# =========================================================
def train_output_name(key):
    lambda_v, occ1, occ2, f = key

    return (
        f"lambda{lambda_v}_"
        f"occ{occ1}.{occ2}_"
        f"f{f}.dat"
    )


def test_output_name(key):
    lambda_v, t1, t2, f = key

    return (
        f"lambda{lambda_v}_"
        f"TEST{t1}_{t2}_"
        f"f{f}.dat"
    )


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# worker
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def process_group(args):
    mode, key, items, input_dir = args

    try:
        # 按 id 排序
        items = sorted(
            items,
            key=lambda x: x[0]
        )

        ids = [x[0] for x in items]

        # 检查完整性
        if ids != VALID_IDS:
            print(f"\n警告：组不完整，跳过")
            print(key)
            print(ids)

            return None

        # 读取图像
        image_list = []

        for _, img_path in items:
            img = load_gray_image(img_path)

            image_list.append(img)

        # SpikeCV simulator
        spikes = images_to_spikes(image_list)

        # 输出文件名
        if mode == "train":
            out_name = train_output_name(key)
        else:
            out_name = test_output_name(key)

        save_path = input_dir / out_name

        # 保存 DAT
        save_dat(spikes, save_path)

        return out_name

    except Exception as e:
        print(f"\n处理失败: {key}")
        print(e)

        return None


# =========================================================
# 处理 train/test
# =========================================================
def process_mode(mode):
    print("\n==============================")
    print(f"处理模式: {mode}")
    print("==============================")

    gt_dir = ROOT_DIR / mode / "gt"
    input_dir = ROOT_DIR / mode / "input"

    input_dir.mkdir(parents=True, exist_ok=True)

    png_files = sorted(gt_dir.glob("*.png"))

    print(f"发现 PNG 数量: {len(png_files)}")

    groups = defaultdict(list)

    # =====================================================
    # 分组
    # =====================================================
    for img_path in png_files:
        filename = img_path.name

        if mode == "train":
            info = parse_train_name(filename)

            if info is None:
                continue

            key = train_group_key(info)

        else:
            info = parse_test_name(filename)

            if info is None:
                continue

            key = test_group_key(info)

        if info["id"] not in VALID_IDS:
            continue

        groups[key].append((info["id"], img_path))

    print(f"发现组数: {len(groups)}")

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++
    # 多进程任务列表
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++
    task_list = []

    for key, items in groups.items():
        task_list.append(
            (
                mode,
                key,
                items,
                input_dir
            )
        )

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++
    # 多进程生成
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++
    print("\n启动多进程...")
    print(f"Worker 数量: {NUM_WORKERS}")

    with ProcessPoolExecutor(
        max_workers=NUM_WORKERS
    ) as executor:
        results = list(
            tqdm(
                executor.map(
                    process_group,
                    task_list
                ),
                total=len(task_list)
            )
        )

    success_count = sum(
        r is not None
        for r in results
    )

    print(f"\n成功生成: {success_count}")

    print(f"\n完成: {mode}")


# =========================================================
# main
# =========================================================
def main():
    print("\n===================================")
    print("SpikeCV DAT Generator")
    print("===================================")

    for mode in MODES:
        process_mode(mode)

    print("\n===================================")
    print("全部 DAT 文件生成完成")
    print("===================================")


# =========================================================
# entry
# =========================================================
if __name__ == "__main__":

    main()
