# =========================================================
# demo_luminance_expansion_multi_thread.py
# 对路演demo用的原始图片进行光度扩充
# 
# 1. 功能：
# 利用 Albumentations库 对 PNG 图像进行：
# (1) 亮度缩放
# (2) 泊松噪声添加
# 
# 2. 解释器要求：
# 安装 albumentations
# 
# 3. 路径要求
# (1) 输入目录：
#     "./demo_dataset_1000x1000/gt"
# (2) 输出目录：
#     "./luminance_expanded_demo_dataset_1000x1000/gt"
# 
# 4. 文件命名格式
# (1) 输入文件：
#     "./demo_dataset_1000x1000/gt"下："[命名，需要读取]_clip[序号1]_f[序号2].png"
# (2) 输出文件：
#     "./luminance_expanded_"./demo_dataset_1000x1000/gt"/gt"下："lambda[光度倍率]_[命名，需要读取]_clip[序号1]_f[序号2].png"
# 
# 5. 采用多线程加速
#
# =========================================================
import os
import cv2
import numpy as np
import albumentations as A
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# 防止 OpenCV 内部线程与 Python 多线程冲突
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++
cv2.setNumThreads(0)


# =========================================================
# 光度倍率
# =========================================================
LAMBDA_LIST = [0.1, 0.3, 0.5, 0.7, 1.0, 2.0]


# =========================================================
# 路径
# =========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

INPUT_DIR = os.path.join(BASE_DIR, "demo_dataset_1000x1000", "gt")

OUTPUT_DIR = os.path.join(BASE_DIR, "luminance_expanded_demo_dataset_1000x1000", "gt")


# =========================================================
# 自定义 Albumentations Transform
# =========================================================
class LuminanceScalingAndPoissonNoise(A.ImageOnlyTransform):
    """
    亮度缩放 + 泊松噪声
    """
    def __init__(self, lam=1.0, always_apply=True, p=1.0):
        super().__init__( always_apply=always_apply,p=p)
        self.lam = lam

    def apply(self, image, **params):
        # =================================================
        # uint8 -> float32 [0,1]
        # =================================================
        img = image.astype(np.float32) / 255.0

        # =================================================
        # 亮度缩放
        # =================================================
        img = img * self.lam

        # =================================================
        # clip
        # =================================================
        img = np.clip(img, 0.0, 1.0)

        # =================================================
        # 泊松噪声
        # =================================================
        poisson_scale = 255.0

        noisy = (
            np.random.poisson(img * poisson_scale) 
            / poisson_scale
        )

        noisy = np.clip(noisy, 0.0, 1.0)

        # =================================================
        # 转回 uint8
        # =================================================
        noisy = (noisy * 255.0).astype(np.uint8)

        return noisy


# =========================================================
# 创建增强 Pipeline
# =========================================================
def build_transform(lam):
    transform = A.Compose([
        LuminanceScalingAndPoissonNoise(lam=lam)
    ])

    return transform


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# 预创建所有 transform
# 避免重复创建 Compose
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++
TRANSFORMS = {
    lam: build_transform(lam)
    for lam in LAMBDA_LIST
}


# =========================================================
# 单张图像处理
# =========================================================
def process_single_image(image_name):
    input_path = os.path.join(INPUT_DIR,image_name)

    # =====================================================
    # 读取图像
    # =====================================================
    image = cv2.imread(input_path, cv2.IMREAD_COLOR)

    if image is None:
        print(f"Failed to read: {input_path}")
        return

    # 去掉后缀
    base_name = os.path.splitext(image_name)[0]

    # =====================================================
    # 对每个 lambda 做增强
    # =====================================================
    for lam in LAMBDA_LIST:
        transform = TRANSFORMS[lam]

        augmented = transform(image=image)

        processed_image = augmented["image"]

        lam_str = str(lam)

        # 输出文件名
        output_name = (f"lambda{lam_str}_"f"{base_name}.png")

        output_path = os.path.join(OUTPUT_DIR, output_name)

        # 保存 PNG
        cv2.imwrite(output_path, processed_image)


# =========================================================
# 主函数
# =========================================================
def main():
    print("=" * 60)
    print("Demo Luminance Expansion Started")
    print("=" * 60)

    # =====================================================
    # 创建输出目录
    # =====================================================
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # =====================================================
    # 获取所有 PNG
    # =====================================================
    image_files = [
        f for f in os.listdir(INPUT_DIR)
        if f.lower().endswith(".png")
    ]

    print(f"\nFound {len(image_files)} images")

    # =====================================================
    # 多线程数量
    # =====================================================
    max_workers = min(
        31,
        os.cpu_count()
    )

    print(f"Using {max_workers} worker threads")

    # =====================================================
    # 多线程执行
    # =====================================================
    with ThreadPoolExecutor(
        max_workers=max_workers
    ) as executor:
        list(
            tqdm(
                executor.map(
                    process_single_image,
                    image_files
                ),
                total=len(image_files)
            )
        )

    print("\nAll Done.")


# =========================================================
# entry
# =========================================================
if __name__ == "__main__":
    main()
