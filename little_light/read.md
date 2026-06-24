你现在接手的是一个 spike camera image reconstruction 项目。我需要你在现有 SwinSF 代码基础上实现“多光度联合训练”的改进版本，不要推倒原模型重写，而是在保持 SwinSF 主体结构兼容的前提下增加 illumination-aware grouped training。

## 背景

我们有一套多光度 spike stream 数据集。对于同一 scene / 同一时间 / 同一 patch，分别生成了多个光度版本：

```python
seen_etas = [0.1, 0.3, 0.5, 0.7, 1.0, 2.0]
```

后续还会构建未见光度测试集：

```python
unseen_etas = [0.2, 0.6, 1.2, 1.5]
```

之前尝试过在 Spk2ImgNet 上做简单多光度扩充，也就是把不同光度样本当成普通独立样本混合训练，但训练不稳定，尤其是低光度下 spike 分布差异很大，容易导致模型优化不稳定。因此这次不要做 naive illumination augmentation，而是显式构建光度维度，让模型学习同一内容在不同光度 spike stream 之间的结构一致性和噪声差异。

## 总目标

请实现一个基于 SwinSF 的 illumination-aware grouped training 框架，核心思想是：

```text
Grouped multi-light spike streams
[B, K, T, H, W]
↓
共享 Spike Feature Extraction
↓
构建光维度特征
[B, K, ...]
↓
Light-Dimension Filtering, LDF
↓
Light-dimension Spike Attention, LSA
↓
SwinSF 原有 spatial-temporal reconstruction backbone
↓
输出每个光度对应的重建图像
```

其中：

* `K` 是光度数量，比如 2 或 6。
* 训练阶段可以使用 grouped multi-light input。
* 测试阶段必须支持 single-light input，也就是 `K=1`，不能强制要求推理时同时输入多个光度版本。
* 原始 SwinSF baseline 必须保留，不能破坏原来的训练和测试流程。

---

# 需要实现的内容

## 1. Dataset：增加两种读取模式

请实现两种 dataset 模式。

### 1.1 single / mixed reading

这个模式把不同光度样本当成独立样本读取，用于 baseline 和普通 eta-conditioned 训练。

返回格式：

```python
inputs: Tensor [T, H, W]
gt:     Tensor [C_gt, H, W] or [1, H, W]
eta:    Tensor scalar or [1]
meta:   dict, optional
```

这个模式允许 DataLoader 随机 shuffle，因此 batch 内 `eta_mean` 可以变化。

### 1.2 grouped illumination reading

这个模式是重点。

每次 `__getitem__` 必须返回同一 scene / 同一时间 / 同一 patch 的多个光度版本：

```python
spikes_group: Tensor [K, T, H, W]
gt:           Tensor [C_gt, H, W] or [1, H, W]
etas:         Tensor [K]
meta:         dict
```

必须保证：

```text
同一 group 内：
- scene_id 一致
- time_id / key frame 一致
- patch 坐标一致
- GT 一致
- 数据增强方式一致
```

如果做随机翻转、旋转、裁剪，同一 group 内所有光度版本必须使用完全相同的增强参数。

请实现 robust grouping 逻辑。推荐通过文件名或 metadata 解析：

```text
scene_id
time_id
patch_id
eta
```

如果当前文件命名规则不统一，请把解析函数写成独立函数，例如：

```python
parse_sample_key(path) -> dict
```

方便我后续修改。

Dataset 参数建议包括：

```python
mode: "single" | "grouped"
etas: list[float]
sample_k: int | None
random_light_subset: bool
return_meta: bool
```

其中：

* `etas` 指定要使用哪些光度。
* `sample_k` 可以从全部光度中随机采样 K 个光度。
* `random_light_subset=True` 时用于 light dropout / 单光度推理适配。
* `sample_k=1` 时 grouped dataset 应该退化为单光度输入。

---

## 2. Model：保留 SwinSF baseline，新增 illumination-aware 版本

不要删除原始 SwinSF 类。请新增一个 wrapper 或新模型类，例如：

```python
class IlluminationAwareSwinSF(nn.Module):
    ...
```

它应该复用原始 SwinSF 的主要模块。

### 2.1 输入兼容

模型需要支持两种输入：

#### single input

```python
spikes: Tensor [B, T, H, W] or [B, 1, T, H, W]
etas:   Tensor [B] or [B, 1]
```

#### grouped input

```python
spikes_group: Tensor [B, K, T, H, W]
etas:         Tensor [B, K]
```

如果 `K=1`，LDF 和 LSA 必须退化为 identity 或近似 identity，保证单光度推理可用。

---

## 3. LDF：Light-Dimension Filtering

请实现一个光维度滤波模块：

```python
class LightDimensionFilter(nn.Module):
    ...
```

输入特征可以是：

```python
F: Tensor [B, K, C, H, W]
```

如果 SwinSF 的 spike feature extraction 输出包含 left / middle / right 三个时间分支，也可以使用：

```python
F: Tensor [B, K, 3, C, H, W]
```

这种情况下请把 LDF 独立作用在每个 temporal branch 上，也就是等价于 reshape 成：

```python
[B * 3, K, C, H, W]
```

再沿 K 维做滤波。

### LDF 设计要求

不要简单平均不同光度特征。请使用残差式、可学习、带门控的滤波：

```python
F_filtered = F + gate * Filter_K(F)
```

建议实现方式：

* 沿光度维 K 做 1D convolution。
* 可以使用 depthwise Conv1d 或 MLP。
* gate 由特征和 / 或 eta embedding 生成。
* 初始化时尽量接近 identity，避免训练初期破坏 SwinSF baseline。

示例思想：

```python
x = rearrange(F, "b k c h w -> (b h w) c k")
delta = conv1d_along_k(x)
delta = rearrange(delta, "(b h w) c k -> b k c h w", ...)
gate = sigmoid(gate_net(...))
F_filtered = F + gate * delta
```

注意不要写死 K=6，K 应该可变。

---

## 4. LSA：Light-dimension Spike Attention

请实现一个光维度注意力模块：

```python
class LightSpikeAttention(nn.Module):
    ...
```

目标：让某个光度的特征可以在同一 scene / patch 的其他光度特征中寻找结构一致信息，从而增强低光特征。

输入：

```python
F: Tensor [B, K, C, H, W]
etas: Tensor [B, K]
```

输出：

```python
F_lsa: Tensor [B, K, C, H, W]
```

如果存在 temporal branch，则同样支持：

```python
[B, K, 3, C, H, W]
```

可通过 reshape 独立处理每个 temporal branch。

### LSA 设计要求

请实现 window-based light attention，避免全图 attention 显存过大。

推荐逻辑：

```text
对每个 spatial window:
    对同一窗口内的 K 个光度特征做 attention
```

可以简化为：

```python
Q = Wq(F_k)
K = Wk(F_all_lights)
V = Wv(F_all_lights)
Attention = softmax(Q @ K^T / sqrt(d))
F_k_enhanced = Attention @ V
```

其中每个 target light 都可以 attend to all lights。

注意：

* attention 是在光度维 K 上进行，而不是替代 SwinSF 原有的 spatial window attention。
* SwinSF 原有 SW-MSA / TSA / RSSB 逻辑要保留。
* LSA 应该插在 Spike Feature Extraction 之后、RSSB/SAB 之前较合适。
* 当 K=1 时，LSA 输出应基本等于输入。

建议加一个 residual scale 参数：

```python
self.lsa_scale = nn.Parameter(torch.zeros(1))
F_out = F + self.lsa_scale * LSA(F)
```

这样初始化时模型接近 baseline。

---

## 5. eta embedding

请实现 eta embedding，用于 LDF/LSA 或 FiLM 调制。

建议：

```python
log_eta = torch.log(eta + 1e-8)
eta_embed = MLP(log_eta)
```

要求：

* 支持 `eta` shape `[B]`, `[B, 1]`, `[B, K]`
* 对 grouped input 输出 `[B, K, D]`
* 不要使用 one-hot 光度类别，因为后续需要 unseen eta 泛化，例如 0.2、0.6、1.2、1.5。

可以先把 eta embedding 接入 LDF gate 和 / 或 LSA 的 Q/K bias。如果实现复杂，先保留接口并在模块内使用简单 FiLM：

```python
F = F * (1 + gamma(eta_embed)) + beta(eta_embed)
```

同样要求最后一层零初始化，使初始状态接近 identity。

---

## 6. Loss：增加 grouped illumination training loss

训练 grouped input 时，模型输出应包含每个光度的重建结果：

```python
preds: Tensor [B, K, C_out, H, W]
```

或者如果 SwinSF 同时输出 left / middle / right 三帧，则按原有格式组织，但至少要能取出 middle reconstruction：

```python
pred_mid: Tensor [B, K, 1, H, W]
```

### 6.1 Reconstruction loss

每个光度都和同一个 GT 监督：

```python
L_rec = sum_k L1(pred_k, gt)
```

如果输出需要按光度归一化，请使用 per-sample eta，而不是 batch eta_mean：

```python
scale = conversion_rate * eta
pred_norm = pred_raw / scale
```

其中 `conversion_rate` 默认 0.6，但必须做成参数：

```python
--conversion_rate 0.6
```

注意 `scale` shape 必须能 broadcast 到 `[B, K, 1, H, W]`，不能使用 `eta.mean()`。

### 6.2 Cross-illumination output consistency loss

同一 group 内不同光度的输出应结构一致。

实现：

```python
L_cons = mean_{i,j} L1(pred_i, pred_j.detach() or pred_j)
```

第一版可以不用 detach，后续可配置。

建议参数：

```python
--lambda_cons 0.05
```

总 loss：

```python
L_total = L_rec + lambda_cons * L_cons
```

### 6.3 可选 feature consistency / distillation

请预留接口，但不必第一版完全实现：

```python
--lambda_feat_cons
--lambda_distill
```

如果实现特征输出，可以返回：

```python
features_dict = {
    "before_ldf": ...,
    "after_ldf": ...,
    "after_lsa": ...
}
```

---

## 7. Single-light inference 约束

非常重要：最终测试和真实应用时通常只有一个光度的 spike stream。

所以测试脚本必须支持：

```python
K = 1
```

例如：

```bash
python test.py --model illum_swinsf --eta 0.1 --single_light
```

此时：

```text
输入: [B, T, H, W] 或 [B, 1, T, H, W]
输出: 单张重建图
```

不能要求同时输入 0.3、0.5、1.0 等其他光度。

可以额外实现 oracle group test：

```bash
python test.py --grouped_eval --etas 0.1 0.3 0.5 0.7 1.0 2.0
```

但这只能作为 upper bound，不是主结果。

---

## 8. Light dropout

为了防止模型训练时过度依赖完整 6 光度输入，请实现 light dropout 或 random light subset。

训练 grouped 模式时，可以随机只保留部分光度：

```python
K_train randomly sampled from {1, 2, 3, 6}
```

或者参数控制：

```python
--light_dropout_prob 0.3
--min_lights 1
```

如果某次只保留 K=1，则 LDF/LSA 应该退化为 identity，loss 只计算 reconstruction loss。

---

## 9. Training script 改造

请在训练脚本中增加参数：

```bash
--model swinsf | illum_swinsf
--dataset_mode single | grouped
--etas 0.1 0.3 0.5 0.7 1.0 2.0
--use_eta_embed
--use_ldf
--use_lsa
--lambda_cons 0.05
--conversion_rate 0.6
--light_dropout_prob 0.0
--min_lights 1
--grad_clip 1.0
```

日志中必须打印：

```text
eta shape
eta unique values
eta group mean
loss_rec
loss_cons
loss_total
PSNR per eta
PSNR average
```

不要只打印 `eta_mean`。如果是 grouped 模式，每个 group 的 eta mean 应该稳定，例如 6 光度时：

```python
mean([0.1,0.3,0.5,0.7,1.0,2.0]) = 0.7667
```

如果只使用 0.1 和 0.3：

```python
mean([0.1,0.3]) = 0.2
```

---

## 10. Evaluation script 改造

测试脚本需要支持：

### 10.1 seen eta test

```bash
--etas 0.1 0.3 0.5 0.7 1.0 2.0
```

### 10.2 unseen eta test

```bash
--etas 0.2 0.6 1.2 1.5
```

### 10.3 single-light inference

每次只输入单个 eta，输出单光度结果。

保存结果时目录包含 eta：

```text
results/eta_0.1/
results/eta_0.2/
...
```

输出指标：

```text
PSNR / SSIM / LPIPS per eta
average over seen etas
average over unseen etas
```

---

## 11. Backward compatibility

必须保证：

1. 原始 SwinSF baseline 仍然可以正常训练和测试。
2. 不使用 `--use_ldf` 和 `--use_lsa` 时，模型应退化到普通 SwinSF 或接近普通 SwinSF。
3. 如果 `dataset_mode=single`，训练流程不能依赖 grouped input。
4. 如果 `K=1`，LDF/LSA 不能报错。
5. 所有新增模块都应有清晰 shape assert 和注释。

---

## 12. 建议实现顺序

请按下面顺序实现，不要一次性写太复杂：

### Step 1

实现 grouped dataset，确认能返回：

```python
spikes_group [B, K, T, H, W]
gt           [B, ...]
etas         [B, K]
```

并打印 eta unique / group mean。

### Step 2

让 SwinSF 支持 grouped input，但先不加 LDF/LSA。做法是 flatten：

```python
[B, K, T, H, W] -> [B*K, T, H, W]
```

共享 SwinSF 前向，再 reshape output 回：

```python
[B, K, ...]
```

实现 `L_rec + L_cons`。

### Step 3

加入 LDF，比较：

```text
Grouped SwinSF
vs
Grouped SwinSF + LDF
```

### Step 4

加入 LSA，比较：

```text
Grouped SwinSF + LDF
vs
Grouped SwinSF + LDF + LSA
```

### Step 5

加入 single-light inference 和 light dropout，确保 K=1 正常运行。

---

## 13. 验收标准

请完成后给出：

1. 修改了哪些文件。
2. 新增了哪些类和函数。
3. 每个新增模块的输入输出 shape。
4. 如何启动 baseline 训练。
5. 如何启动 grouped illumination training。
6. 如何测试 seen eta。
7. 如何测试 unseen eta。
8. 一个最小 smoke test，至少验证：

   * grouped dataset 输出 shape 正确；
   * K=6 前向正常；
   * K=1 前向正常；
   * loss 可以 backward；
   * 原始 SwinSF baseline 不受影响。

请尽量保持代码清晰、模块化，不要把所有逻辑堆进 train.py。
