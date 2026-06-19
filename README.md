---

# Luminance-Expanded-SpikeImg-Reconstruction
《极微光环境下脉冲相机图象的精确重建》项目文件

---

# 目录
## 一、 dataset processing prog: 数据集处理程序（by:Shadowindows_017）
###  (i) luminance_expansion_multi_thread.py
1. 功能：

利用 Albumentations库 对 PNG 图像进行：

(1) 亮度缩放

(2) 泊松噪声添加

2. 解释器要求：

安装 albumentations

3. 路径要求

(1) 输入目录：

        "./spike_x4k/train/gt"

        "./spike_x4k/test/gt"

(2) 输出目录：

        "./luminance_expanded_spike_x4k/train/gt"

        "./luminance_expanded_spike_x4k/test/gt"

4. 文件命名格式

(1) 输入文件：

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
        (1) 安装北大官方的Spike库
        安装过程（在anaconda环境命令行中依次运行）：
            git clone https://git.openi.org.cn/Cordium/SpikeCV.git （一台pc安装一次SpikeCV即可，剩下的需要每个环境执行一次）
            cd SpikeCV （安装好的SpikeCV是个文件夹，进入该目录执行下一条命令）
            pip install -e . （文件夹里预存了配置信息，运行这条命令自动安装配置，但是numpy版本不对需要手动改）
        （安装后实际调用的时候，程序报错区可能还会显示“无法解析导入Spike”，无视掉，正常跑就行）
        (2) pyhton版本为3.10.x
        (3) 安装numpy v1.26.x
        
        3. 时间结构：
        (1) 全0空白timestep
        (2) 一组5张图，每张图生成8个timestep
        (3) 一个.dat文件共含41timestep
        
        4. 路径要求
        (1) 输入目录：
            "./luminance_expanded_spike_x4k/train/gt",
            "./luminance_expanded_spike_x4k/test/gt"
        (2) 输出目录：
            "./luminance_expanded_spike_x4k/train/input"
            "./luminance_expanded_spike_x4k/test/input"
            
        5. 文件命名格式
        (1) 输入文件：
            "./luminance_expanded_spike_x4k/train/gt"下："lambda[光度倍率]_occ[序号1].[序号2]_f[序号3]_key_id[id号].png"
            "./luminance_expanded_spike_x4k/test/gt"下："lambda[光度倍率]_TEST[序号1]_[序号2]_f[序号3]_key_id[id号].png"
        (2) 输出文件：
            "./luminance_expanded_spike_x4k/train/input"下："lambda[光度倍率]_occ[序号1].[序号2]_f[序号3].dat"
            "./luminance_expanded_spike_x4k/test/input"下："lambda[光度倍率]_TEST[序号1]_[序号2]_f[序号3].dat"
            
        6. 注意：
        本程序按需求对SpikeCV\SpikeCV\spkData\convert_img.py中的img_to_spike函数做了修改，以适配多线程加速。
        (目录例：本机地址为C:\Users\Lenovo\SpikeCV\SpikeCV\spkData\convert_img.py)
        若后续需要使用原本的功能，请记得改回文件。
        
        7. 采用多线程加速
   
###  (iii) demo_luminance_expansion_multi_thread.py: 对路演demo用的原始图片进行光度扩充
        1. 功能：
        利用 Albumentations库 对 PNG 图像进行：
        (1) 亮度缩放
        (2) 泊松噪声添加

        2. 解释器要求：
        安装 albumentations

        3. 路径要求
        (1) 输入目录：
            "./demo_dataset_1000x1000/gt"
        (2) 输出目录：
            "./luminance_expanded_demo_dataset_1000x1000/gt"

        4. 文件命名格式
        (1) 输入文件：
            "./demo_dataset_1000x1000/gt"下："[命名，需要读取]_clip[序号1]_f[序号2].png"
        (2) 输出文件：
            "./luminance_expanded_"./demo_dataset_1000x1000/gt"/gt"下："lambda[光度倍率]_[命名，需要读取]_clip[序号1]_f[序号2].png"

        5. 采用多线程加速
   
###  (iv) demo_spike_generator_multi_thread.py: 用路演demo图片生成.dat脉冲流文件
        1. 功能
        使用官方 Spike库 的 img_to_spike函数 将.png图片组转换为.dat脉冲流文件

        2. 解释器要求：
        (1) 安装北大官方的Spike库
        安装过程（在anaconda环境命令行中依次运行）：
            git clone https://git.openi.org.cn/Cordium/SpikeCV.git （一台pc安装一次SpikeCV即可，剩下的需要每个环境执行一次）
            cd SpikeCV （安装好的SpikeCV是个文件夹，进入该目录执行下一条命令）
            pip install -e . （文件夹里预存了配置信息，运行这条命令自动安装配置，但是numpy版本不对需要手动改）
        （安装后实际调用的时候，程序报错区可能还会显示“无法解析导入Spike”，无视掉，正常跑就行）
        (2) pyhton版本为3.10.x
        (3) 安装numpy v1.26.x

        3. 时间结构：
        (1) 全0空白timestep
        (2) 一组5张图，每张图生成8个timestep
        (3) 一个.dat文件共含41timestep

        4. 路径要求
        (1) 输入目录：
            "./luminance_expanded_demo_dataset_1000x1000/gt"
        (2) 输出目录：
            "./luminance_expanded_demo_dataset_1000x1000/input"

        5. 文件命名格式
        (1) 输入文件：
            "./luminance_expanded_demo_dataset_1000x1000/gt"下："lambda[光度倍率]_[命名，需要读取]_clip[序号1]_f[序号2].png"
        (2) 输出文件：
            "./luminance_expanded_demo_dataset_1000x1000/input"下："lambda[光度倍率]_[命名，需要读取]_clip[序号1].dat"

        6. 注意：
        本程序按需求对SpikeCV\SpikeCV\spkData\convert_img.py中的img_to_spike函数做了修改，以适配多线程加速。
        (目录例：本机地址为C:\Users\Lenovo\SpikeCV\SpikeCV\spkData\convert_img.py)
        若后续需要使用原本的功能，请记得改回文件。

        7. 采用多线程加速

###  (v) spike_reader.py: 可视化.dat脉冲流文件
       1. 功能
       可视化查看脉冲流文件
       
       2. 注意：
       (1) 一次只能看一个文件
       (2) 需要手动改文件名："# DAT 文件路径"下的DAT_PATH
       (3) "# 参数"下的HEIGHT、WIDTH、TIMESTEPS需要与脉冲流生成参数一致

###  (vi) convert_img.py: 修改后的同名库文件

## 二、模型训练文件
(待编写)
