# Extreme Low-Light Spike Camera Image Reconstruction

本项目面向**脉冲相机极微光图像重建任务**，研究如何利用同一场景在不同光度下的 spike stream 构建“光度维度”，从而提升弱光、极弱光以及跨光度条件下的图像重建质量。

我们基于 SwinSF 进行改进，提出轻量化光度感知模型 **LA-SwinSF-Lite**。该方法在训练阶段利用多光度 grouped 数据学习跨光度规律，并通过知识蒸馏使学生模型在推理阶段仅需单一光度 spike stream 输入。

---

## Project Overview

脉冲相机通过“积分-触发-重置”的方式连续记录光强变化。每个像素持续累积入射光，当累计值达到阈值时产生一个 spike。相比传统相机，脉冲相机具有更高的时间分辨率和动态范围，但其输出是二进制 spike stream，不能直接作为图像使用。

在极弱光环境下，spike 发放频率明显降低，脉冲流变得稀疏且不稳定，导致图像重建难度显著增加。

本项目关注的问题是：

> 如何让模型学习 spike stream 随光度变化的统计规律，从而提升极微光条件下的图像重建质量？

---

## Motivation

我们首先尝试将不同光度的数据作为普通数据增强直接混合训练，但实验表明这种方式并不稳定，甚至会导致性能下降。

主要原因包括：

- 不同光度下 spike density 差异明显；
- 弱光 spike 更稀疏，噪声占比更高；
- 简单混合训练会造成不同光度分布之间的冲突；
- 模型无法显式理解“同一场景不同光度版本”之间的结构对应关系。

因此，多光度数据的价值不在于简单增加样本数量，而在于提供同一场景下跨光度的结构一致性和光度变化规律。

---

## Main Contributions

- 构建多光度 spike 数据集，对同一组图像生成多个光度版本；
- 验证简单多光度数据增强会引入负迁移；
- 提出 **LA-SwinSF-Lite**，以较低显存开销建模光度维度；
- 设计 Light Descriptor、LDF-Lite、LSA-Lite 和 Light Adapter；
- 引入知识蒸馏，使最终模型在推理阶段只需要单光度输入；
- 在弱光重建、跨光度推理和真实数据实验中验证方法的有效性。

---

## Usage

```bash
# 训练教师模型
bash train.sh

# 训练学生模型
bash dis.py

# 测试教师模型
bash test-t.sh

# 测试学生模型
bash test-s.sh

# 真实数据测试
bash make-real.sh
```

---

## Dataset

本项目基于 spike-x4k 构建多光度 spike 数据集。

训练光度包括：

```text
0.1, 0.3, 0.5, 0.7, 1.0, 2.0
```

每个光度版本对应同一场景、同一时间和同一图像内容，仅改变光度强度，并通过 SpikeCV 转换为 `.dat` 格式的二进制 spike stream。

未见光度测试可进一步使用：

```text
0.2, 0.6, 1.2, 1.5
```

数据构建目标是让模型不仅记住固定光度档位，还能够学习连续光度变化下 spike stream 的统计规律。

---

## Method

### LA-SwinSF-Lite

原始多光度输入形式为：

```text
S_group: [B, K, T, H, W]
etas:    [B, K]
```

其中：

- `B` 表示 batch size；
- `K` 表示光度数量；
- `T` 表示 spike 时间帧数；
- `H, W` 表示空间尺寸。

如果直接将所有光度输入 SwinSF 主干，显存开销会随着 `K` 成倍增加。因此，LA-SwinSF-Lite 采用轻量化策略：

> 只让目标光度 spike stream 进入 SwinSF 主干，其余光度被压缩为轻量光度描述符，用于指导目标光度重建。

整体流程为：

```text
Grouped multi-light spike streams
        -> Light Descriptor Encoder
        -> LDF-Lite
        -> LSA-Lite
        -> Target light code
        -> Light Adapter
        -> SwinSF Backbone
        -> Reconstructed image
```

### Light Descriptor Encoder

Light Descriptor Encoder 将每个光度的大规模 spike stream 压缩为一个低维描述符：

```text
[B, K, T, H, W] -> [B, K, D]
```

其中 `D` 是 descriptor 维度。描述符包含的信息包括：

- 光度因子 `eta`；
- 平均 spike 发放率；
- spike 发放率标准差；
- 时间维度 spike profile；
- 低分辨率 spike density map。

该模块将多光度输入中的统计信息转化为轻量光度提示，避免将所有光度数据送入重型主干。

### LDF-Lite

LDF-Lite 表示 **Light-Dimension Filtering**。它在光度维度上对 descriptor 进行轻量滤波，使弱光描述更加稳定：

```text
D' = D + gate * Filter_K(D)
```

其中：

- `Filter_K` 沿光度维度进行滤波；
- `gate` 控制参考其他光度信息的强度；
- 当 `K=1` 时，LDF-Lite 退化为恒等映射。

### LSA-Lite

LSA-Lite 表示 **Light-dimension Spike Attention**。它让目标光度从其他光度 descriptor 中选择有用信息：

```text
Query:  target light descriptor
Key:    all light descriptors
Value:  all light descriptors
```

注意力只发生在 `[B, K, D]` 的轻量描述符上，而不是在 `[B, K, C, H, W]` 的大型特征图上进行，因此显存开销较小。

### Light Adapter

经过 LDF-Lite 和 LSA-Lite 后，模型得到目标光度编码 `z_target`。Light Adapter 将该编码转化为 SwinSF 特征调制参数：

```text
gamma, beta = Adapter(z_target)
F' = F * (1 + gamma) + beta
```

其中 `F` 是 SwinSF 主干中的中间特征。该设计使模型能够根据当前光度状态自适应调整特征表达。

---

## Knowledge Distillation

为了让最终模型在真实推理时只依赖单光度输入，本项目引入 teacher-student 知识蒸馏。

### Teacher

Teacher 使用 grouped 多光度输入，能够利用同一场景下不同光度版本提供的结构一致性和光度统计信息。

```text
S_group -> Teacher -> pred_teacher
```

### Student

Student 只输入目标光度 spike stream，例如：

```text
S_0.3 -> Student -> pred_student
```

推理阶段最终使用 Student，因此不需要同时输入多个光度版本。

### Distillation Objective

蒸馏训练包含重建损失和蒸馏损失：

```text
L = L_rec + lambda_distill * L_distill
```

其中：

```text
L_rec     = ||pred_student - GT||_1
L_distill = ||pred_student - stopgrad(pred_teacher)||_1
```

Teacher 在蒸馏阶段保持冻结，不参与反向传播。Student 在学习 GT 监督的同时模仿 Teacher 的输出，从而继承 grouped 多光度训练带来的结构先验。

---

## Experimental Observations

实验结果表明，弱光重建难度显著高于正常光照重建。

| Training Setting | Test Eta | PSNR | SSIM | Observation |
|---|---:|---:|---:|---|
| Single-light training | 0.3 | 28.11 | 0.692 | Weak-light baseline |
| Single-light training | 1.0 | 39.61 | 0.968 | Normal-light reference |
| Naive all-light augmentation | 0.3 | 27.32 | 0.616 | Negative transfer |
| LA-SwinSF-Lite grouped training | 0.3 | 36.12 | 0.926 | Effective light-dimension modeling |

主要结论：

- `eta=0.3` 的重建性能明显低于 `eta=1.0`，说明弱光 spike 重建本身难度较大；
- 全光度简单混合训练低于单光度 `0.3`，说明多光度数据不能直接作为普通数据增强；
- grouped 联合训练明显优于单光度训练和简单增强，说明光度维度建模有效。

---

## Ablation Study

| Setting | Descriptor | LDF-Lite | LSA-Lite | Distillation | Purpose |
|---|---|---|---|---|---|
| Baseline | - | - | - | - | Original SwinSF baseline |
| + Descriptor | Yes | - | - | - | Test light statistic guidance |
| + LDF-Lite | Yes | Yes | - | - | Test light-dimension filtering |
| + LSA-Lite | Yes | - | Yes | - | Test light attention |
| + LDF + LSA | Yes | Yes | Yes | - | Test combined light modeling |
| Full Model | Yes | Yes | Yes | Yes | Test full distillation framework |

---

## Cross-Light Inference

为了验证模型是否学习到跨光度规律，可以设计不同光度之间的推理模式：

| Mode | Description |
|---|---|
| 0.5 -> 0.3 | 使用较高弱光光度提示，推理更低光度目标 |
| 0.3 -> 0.5 | 使用低光光度提示，推理较高光度目标 |
| 1.0 -> 0.3 | 使用正常光照提示，推理弱光目标 |

该实验用于观察模型在光度变化下的泛化能力，以及光度提示对重建结果的影响。

---

## Real Data Inference

真实数据通常没有明确的合成光度 `eta`，因此真实数据推理时可以采用以下策略：

- 固定使用弱光提示，例如 `eta=0.3`；
- 对多个 `eta` 进行 sweep，比较输出的视觉质量；
- 根据真实 spike stream 的平均发放率估计伪 `eta`；
- 对输出进行自动亮度拉伸，仅作为可视化辅助。

真实数据实验主要用于验证模型在非合成环境下的可用性和泛化潜力。

---

## Repository Structure

```text
.
├── train.sh
├── dis.py
├── test-t.sh
├── test-s.sh
├── make-real.sh
├── dataloader.py
├── train.py
├── train_lite_distill.py
├── test_fixed.py
├── test_lite_real_fixed.py
├── models/
│   ├── SwinSF_1000.py
│   └── SwinSF_250.py
├── checkpoint/
├── save/
└── README.md
```

---

## Notes

- 多光度数据不应被简单视为普通数据增强；
- grouped training 的核心是利用同一场景的跨光度结构一致性；
- LA-SwinSF-Lite 避免了全光度输入 SwinSF 主干造成的显存开销；
- 知识蒸馏使最终模型能够在单光度输入下完成推理；
- 真实数据中的 `eta` 不是严格物理量，更适合作为弱光条件提示。

---

## References

```bibtex
@inproceedings{zhao2021spk2imgnet,
  title={Spk2ImgNet: Learning to Reconstruct Dynamic Scene from Continuous Spike Stream},
  author={Zhao, Jing and Xiong, Rui and Liu, Hang and Zhang, Jian and Huang, Tiejun},
  booktitle={CVPR},
  year={2021}
}
```

```bibtex
@inproceedings{zhao2024bsf,
  title={Boosting Spike Camera Image Reconstruction from a Perspective of Dealing with Spike Fluctuations},
  author={Zhao, Rui and Xiong, Rui and Zhao, Jing and Zhang, Jian and Fan, Xin and Yu, Zhaofei and Huang, Tiejun},
  booktitle={CVPR},
  year={2024}
}
```

```bibtex
@article{jiang2024swinsf,
  title={SwinSF: Image Reconstruction from Spatial-Temporal Spike Streams},
  author={Jiang, L. and Zhu, C. and Chen, Y.},
  journal={arXiv preprint arXiv:2407.15708},
  year={2024}
}
```

---

## Acknowledgement

This project is developed for research and course project purposes. It is inspired by existing spike camera image reconstruction works including Spk2ImgNet, BSF, and SwinSF.


---

# 文件目录
## 一、 dataset processing prog: 数据集处理程序
###  (i) luminance_expansion_multi_thread.py
1. 功能：

利用 Albumentations库 对 PNG 图像进行：

（1）亮度缩放

（2）泊松噪声添加

2. 解释器要求：

安装 albumentations

3. 路径要求

(1) 输入目录：

        "./spike_x4k/train/gt"
        "./spike_x4k/test/gt"

（2）输出目录：

        "./luminance_expanded_spike_x4k/train/gt"
        "./luminance_expanded_spike_x4k/test/gt"

4. 文件命名格式

（1）输入文件：

        "./spike_x4k/train/gt"下："occ[序号1].[序号2]_f[序号3]_key_id[id号].png"
        "./spike_x4k/test/gt"下："TEST[序号1]_[序号2]_f[序号3]_key_id[id号].png"

(2) 输出文件：

        "./luminance_expanded_spike_x4k/train/gt"下："lambda[光度倍率]_occ[序号1].[序号2]_f[序号3]_key_id[id号].png"
        "./luminance_expanded_spike_x4k/test/gt"下："lambda[光度倍率]_TEST[序号1]_[序号2]_f[序号3]_key_id[id号].png"

5. 采用多线程加速
        
###  (ii) spike_generator_multi_thread.py: 用数据集图片生成.dat脉冲流文件
1. 功能

使用官方 Spike库 的 img_to_spike函数 将.png图片组转换为.dat脉冲流文件

2. 解释器要求：

（1）安装北大官方的Spike库

安装过程（在anaconda环境命令行中依次运行）：
       
        git clone https://git.openi.org.cn/Cordium/SpikeCV.git （一台pc安装一次SpikeCV即可，剩下的需要每个环境执行一次）
        cd SpikeCV （安装好的SpikeCV是个文件夹，进入该目录执行下一条命令）
        pip install -e . （文件夹里预存了配置信息，运行这条命令自动安装配置，但是numpy版本不对需要手动改）

（安装后实际调用的时候，程序报错区可能还会显示“无法解析导入Spike”，无视掉，正常跑就行）

（2）pyhton版本为 3.10.x

（3）numpy版本为 1.26.x

3. 时间结构：

（1）全0空白timestep

（2）一组5张图，每张图生成8个timestep

（3）一个.dat文件共含41timestep

4. 路径要求

（1）输入目录：

        "./luminance_expanded_spike_x4k/train/gt",
        "./luminance_expanded_spike_x4k/test/gt"

（2）输出目录：

        "./luminance_expanded_spike_x4k/train/input"
        "./luminance_expanded_spike_x4k/test/input"
  
5. 文件命名格式

（1）输入文件：
        
        "./luminance_expanded_spike_x4k/train/gt"下："lambda[光度倍率]_occ[序号1].[序号2]_f[序号3]_key_id[id号].png"
        "./luminance_expanded_spike_x4k/test/gt"下："lambda[光度倍率]_TEST[序号1]_[序号2]_f[序号3]_key_id[id号].png"

（2）输出文件：

        "./luminance_expanded_spike_x4k/train/input"下："lambda[光度倍率]_occ[序号1].[序号2]_f[序号3].dat"
        "./luminance_expanded_spike_x4k/test/input"下："lambda[光度倍率]_TEST[序号1]_[序号2]_f[序号3].dat"
   
7. 注意：

本程序按需求对

        SpikeCV\SpikeCV\spkData\convert_img.py

中的

        img_to_spike()

做了修改，以适配多线程。

(目录例：本机地址)

        C:\Users\Lenovo\SpikeCV\SpikeCV\spkData\convert_img.py

若后续需要使用原本的功能，请记得改回文件。
        
7. 采用多线程加速
   
###  (iii) demo_luminance_expansion_multi_thread.py: 对路演demo用的原始图片进行光度扩充
1. 功能：

利用 Albumentations库 对 PNG 图像进行：

（1）亮度缩放

（2）泊松噪声添加

2. 解释器要求：

安装 albumentations

3. 路径要求

（1）输入目录：

        "./demo_dataset_1000x1000/gt"

（2）输出目录：

        "./luminance_expanded_demo_dataset_1000x1000/gt"

4. 文件命名格式

（1）输入文件：

        "./demo_dataset_1000x1000/gt"下："[命名，需要读取]_clip[序号1]_f[序号2].png"

（2）输出文件：

        "./luminance_expanded_"./demo_dataset_1000x1000/gt"/gt"下："lambda[光度倍率]_[命名，需要读取]_clip[序号1]_f[序号2].png"

5. 采用多线程加速
   
###  (iv) demo_spike_generator_multi_thread.py: 用路演demo图片生成.dat脉冲流文件
1. 功能

使用官方 Spike库 的 img_to_spike函数 将.png图片组转换为.dat脉冲流文件

2. 解释器要求：

（1）安装北大官方的Spike库

安装过程（在anaconda环境命令行中依次运行）：

        git clone https://git.openi.org.cn/Cordium/SpikeCV.git （一台pc安装一次SpikeCV即可，剩下的需要每个环境执行一次）
        cd SpikeCV （安装好的SpikeCV是个文件夹，进入该目录执行下一条命令）
        pip install -e . （文件夹里预存了配置信息，运行这条命令自动安装配置，但是numpy版本不对需要手动改）

（安装后实际调用的时候，程序报错区可能还会显示“无法解析导入Spike”，无视掉，正常跑就行）

（2）pyhton版本为 3.10.x

（3）numpy版本为 1.26.x

3. 时间结构：

（1）全0空白timestep

（2）一组5张图，每张图生成8个timestep

（3）一个.dat文件共含41timestep

4. 路径要求

（1）输入目录：

        "./luminance_expanded_demo_dataset_1000x1000/gt"

（2）输出目录：

        "./luminance_expanded_demo_dataset_1000x1000/input"

6. 文件命名格式

（1）输入文件：

        "./luminance_expanded_demo_dataset_1000x1000/gt"下："lambda[光度倍率]_[命名，需要读取]_clip[序号1]_f[序号2].png"

（2）输出文件：

        "./luminance_expanded_demo_dataset_1000x1000/input"下："lambda[光度倍率]_[命名，需要读取]_clip[序号1].dat"

6. 注意：

本程序按需求对

        SpikeCV\SpikeCV\spkData\convert_img.py

中的

        img_to_spike()

做了修改，以适配多线程。

(目录例：本机地址)

        C:\Users\Lenovo\SpikeCV\SpikeCV\spkData\convert_img.py

若后续需要使用原本的功能，请记得改回文件。

7. 采用多线程加速

###  (v) spike_reader.py: 可视化.dat脉冲流文件

1. 功能

可视化查看脉冲流文件
       
2. 注意：

（1）一次只能看一个文件

（2）需要手动改文件名："# DAT 文件路径"下的DAT_PATH

（3）"# 参数"下的HEIGHT、WIDTH、TIMESTEPS需要与脉冲流生成时的同名参数一致

###  (vi) convert_img.py: 修改后的同名库文件

## 二、little_light：模型&训练文件（by：陈诺）

详情见文件夹内"read.md"
