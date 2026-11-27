# 用 NumPy 复现 OpenCLIP ViT-B/32 图像编码器（小项目一）

只用 **NumPy** 实现 OpenCLIP **ViT-B-32** 视觉塔的完整前向计算：输入一批预处理好的
图像，输出每张图一个 512 维、单位长度的特征向量。模型参数（152 个 `visual.*` 张量，
共 87,849,216 个 float32 值）由课程提供，本项目**不做任何训练/梯度更新**，只复现推理。

数值上要求与独立 Torch FP32 前向及期末项目的教师参考完全对齐：

| 指标 | 容差 | 本项目实测（10 张图） |
| --- | --- | --- |
| 输出 token `[B,512]` | atol 2e-5 / rtol 2e-4 | **max_abs = 5.4e-07** PASS |
| 内部 token `[B,50,768]` | atol 2e-4 / rtol 2e-4 | **max_abs = 4.8e-05** PASS |
| patch 数组 `[B,49,3072]` | atol 2e-4 | **max_abs = 0** PASS |
| 预处理 `[B,3,224,224]` | atol 2e-6 | **max_abs = 0** PASS |

完整的分阶段误差、耗时与对照实验见 [docs/metrics.json](docs/metrics.json) 与实验报告
[docs/实验报告.docx](docs/实验报告.docx)。

## 功能与结构

数据流（`B` 为批量大小）：

```
[B,3,224,224] --patchify--> [B,49,3072] --线性投影(无 bias)--> [B,49,768]
      --拼 CLS--> [B,50,768] --加位置编码--> [B,50,768] --ln_pre-->
      12 × TransformerBlock --> [B,50,768] --取 CLS--> [B,768] --ln_post-->
      [B,768] --@ visual.proj--> [B,512] --L2 归一化--> [B,512]
```

单个 `TransformerBlock`：

```
x --ln1--> --QKV 线性--> [B,50,2304] --切成 12 头--> q,k,v [B,12,50,64]
  --scores=q·kᵀ/√64--> [B,12,50,50] --softmax--> --·v--> [B,12,50,64]
  --合并头--> [B,50,768] --out_proj--> + x  (第一处残差)
  --ln2--> --c_fc--> [B,50,3072] --GELU(erf)--> --c_proj--> + residual (第二处残差)
```

| 文件 | 职责 |
| --- | --- |
| `image_encoder.py` | **本项目实现的唯一文件**：`MultiHeadAttention`、`TransformerBlock`、`ImageEncoder`（六个 TODO 均在此） |
| `helpers.py` | 课程提供：`LayerNorm`、精确 `GELU`(erf)、`attention_weights`(softmax)、`causal_mask`、`unit_vector` |
| `preprocess_image.py` | 课程提供：短边缩放到 224、中心裁剪、CHW 转换与 OpenCLIP 均值方差归一化 |
| `runtime.py` | 课程提供：按包内相对路径读取 `config.json`、权重、10 张图片 |
| `check.py` | 课程提供：8 个阶段的数值自检（assets/preprocess/patches/tokens/block0/last_tokens/output/batch） |
| `run.py` | 课程提供：命令行推理入口，输出 `output_tokens.npz` |
| `config.json` | 课程提供：固定 checkpoint 的结构超参（224/32/768/12 头/12 层/512 维） |
| `assets/` | 10 张课程图片（`images.json` 含来源与 SHA-256）与 `reference_outputs.npz`（仅用于自检） |
| `tools/report_metrics.py` | 本项目的证据脚本：分阶段误差、耗时、批量一致性、6 组错误实现对照实验、两张报告插图 |
| `docs/` | 实验报告（docx/pdf）、`metrics.json`、插图 |
| `docs/latex/` | 课程 LaTeX 模板版的实验报告源码（Overleaf 选 **XeLaTeX** 编译，`main.tex` + `figs/`） |

六个实现任务（全部在 `image_encoder.py`）：

| TODO | 位置 | 内容 |
| --- | --- | --- |
| 01 | `ImageEncoder.patchify` | `[B,3,224,224] → [B,49,3072]`，patch 从左到右、从上到下 |
| 02 | `MultiHeadAttention.split_heads` | `[B,N,768] → [B,12,N,64]` |
| 03 | `MultiHeadAttention.__call__` | 缩放点积注意力、加权求和、拼回头、输出投影 |
| 04 | `TransformerBlock.__call__` | 两处残差 + MLP（`GELU(u·W1ᵀ+b1)·W2ᵀ+b2`） |
| 05 | `ImageEncoder.embed` | patch 投影、CLS 拼接、位置编码相加 |
| 06 | `ImageEncoder.__call__` | 取 CLS、`ln_post`、`visual.proj` 投影、单位化 |

## 环境与依赖

- Python ≥ 3.9（实测 3.12）
- `numpy>=1.24,<3`、`Pillow>=9.2,<13`（见 `requirements.txt`）
- 不需要 GPU、PyTorch、OpenCLIP 或联网推理；纯 CPU，10 张图约 3.2 s

```bash
python -m pip install -r requirements.txt
```

## 必要数据与获取方法

仓库**不包含**两个大文件（单个 326 MB，超过 GitHub 单文件 100 MB 限制，且课程要求
不要重复提交提供的参数/参考数据）：

| 文件 | 大小 | SHA-256 | 获取方式 |
| --- | --- | --- | --- |
| `weights/image_encoder.npz` | 326 MB | `8567f805696fb37412e343de59dee7e677f35e77d6ef297fbdd6fa7b1923a740` | 从课程附件 `project1-学生版.zip` 解压后复制到 `weights/` |
| `assets/reference_outputs.npz` | 7.6 MB | `ef07334da5d2b1f2f0f1e118c7ad34e2e68a69b9ad9bdf9be505492e12ae3f71` | 同上，复制到 `assets/` |

只需要运行 `run.py` 而不做自检时，`assets/reference_outputs.npz` 不是必需的；
`weights/manifest.json` 里记录了 152 个张量各自的形状、dtype 与 SHA-256，可用于校验下载完整性。

## 运行

在仓库根目录执行：

```bash
# 1) 分阶段数值自检（按顺序排查第一个出错的阶段）
python check.py --stage patches
python check.py --stage tokens
python check.py --stage block0
python check.py --stage last_tokens
python check.py --stage output
python check.py --stage batch
python check.py --stage all        # 一次跑完 8 个阶段

# 2) 推理：10 张课程图片 -> output_tokens.npz
python run.py --output output_tokens.npz

# 3) 任意自己的图片（顺序与 ids 一致）
python run.py --images path/to/a.jpg path/to/b.png --output my_tokens.npz

# 4) 复现报告里的全部证据（误差表、耗时、对照实验、插图）
python tools/report_metrics.py
```

`check.py` 的判定条件：形状一致、`float32`、全为有限值、且
`|actual - reference| <= atol + rtol*|reference|`；输出阶段额外校验每行 L2 范数为 1。

## 关键实现约定（容易踩坑的地方）

1. 每个 patch 内部按 **C,H,W** 展平，即 `conv1.weight.reshape(768,3072)` 的顺序；patch
   之间按“先左右、后上下”排列。
2. patch 投影**没有 bias**，`conv1.weight` 已重排为 `[768,3072]`，使用 `patches @ W.T`。
3. 注意力/MLP 的线性层权重按 `[out,in]` 存放，必须用 `W.T`；只有 `visual.proj` 是
   `[in,out]`，**不能**转置。
4. `k` 是 4D `[B,H,N,64]`，只能用 `transpose(0,1,3,2)` 交换最后两轴；`k.T` 会把 4 个轴
   全部反转（见报告中的对照实验，会直接抛 `ValueError` 或静默算错）。
5. `split_heads` 按 **连续 64 维** 切分，不能按步长交错取；合并时是
   `transpose(0,2,1,3).reshape(B,N,768)`。
6. 全程保持 `float32`：除法用 `np.float32(np.sqrt(64))`，避免与 `float64` 标量运算时
   被提升为 `float64`。
7. 图像塔**没有** LayerScale，块内只有两处残差；`causal` 仅给后续文本塔复用。

## 复现证据

`tools/report_metrics.py` 输出 `docs/metrics.json`（分阶段最大/平均绝对误差、耗时、
输出范数、批量重排/重复一致性、6 组错误实现对照）和两张插图
（`docs/figures/stage_error.png`、`docs/figures/patch_order.png`）。实测：

- 输出 token 单位长度：`1.000000 ± 7e-8`，任意两图余弦相似度 0.462–0.674（特征不塌缩）
- 批次顺序 `[1,0,1]`（含重复图）与逐张推理一致，最大差 4.5e-07
- 错误实现对照（block0 最大误差）：注意力投影漏转置 1.88、漏位置编码 1.78、
  合并头漏转置 1.91、交错切分头 0.87、误用因果掩码 0.44，而正确实现为 7.1e-06

## 已知局限

- `GELU` 是课程提供的 Python 逐元素 `math.erf` 循环，是整条管线的主要耗时来源
  （12 层约 3 s/10 张图）；可改用向量化 `erf` 近似加速，但会改变最后几位数值。
- 输入固定 224×224、patch 32，代码未支持任意分辨率；批量维度可任意。
- 精度上限由 float32 累加顺序决定，与 Torch 的差异约 1e-5 量级，属正常浮点差异。

## 实验报告

[docs/实验报告.docx](docs/实验报告.docx) / [docs/实验报告.pdf](docs/实验报告.pdf)：
任务与方法、本人贡献、结果证据（分阶段误差、插图、对照实验）、分析与反思（含一个
形状变化与一个 bug 修复的详细说明）。正文控制在 5 页以内。
