# 视觉注意力汇聚现象复现指南

本指南说明如何基于论文"SEE WHAT YOU ARE TOLD: VISUAL ATTENTION SINK IN LARGE MULTIMODAL MODELS"复现LLaVA-1.5-7B模型的实验结果，并对模型的visual attention sink现象进行可视化。

## 目录

1. [项目概述](#项目概述)
2. [环境配置](#环境配置)
3. [数据准备](#数据准备)
4. [运行实验](#运行实验)
5. [可视化分析](#可视化分析)
6. [结果解读](#结果解读)

---

## 项目概述

### 什么是视觉注意力汇聚（Visual Attention Sink）？

在大型多模态模型中，研究发现某些视觉token会不成比例地接收大量注意力，成为"注意力汇聚点"（attention sinks）。这些汇聚点可能：
- 不具有语义相关性
- 导致模型对其他重要视觉特征的注意力降低
- 引发目标幻觉（object hallucination）问题

### 本项目提供的功能

1. **注意力捕获**：在模型推理过程中捕获注意力权重
2. **可视化工具**：生成注意力热图和汇聚点分析图
3. **VAR机制**：实现论文提出的视觉注意力重分配机制
4. **实验复现**：提供完整的实验配置和脚本

---

## 环境配置

### 1. 创建Conda环境

```bash
conda create -n VisAttnSink python=3.11
conda activate VisAttnSink
```

### 2. 安装依赖

```bash
# 安装PyTorch (根据你的CUDA版本调整)
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# 安装其他依赖
pip install -r env_pip.txt
```

### 3. 下载LLaVA-1.5-7B模型

```bash
# 使用HuggingFace CLI
huggingface-cli download liuhaotian/llava-v1.5-7b --local-dir /path/to/llava-v1.5-7b

# 或在配置文件中直接使用HuggingFace Hub路径
# path_model: liuhaotian/llava-v1.5-7b
```

---

## 数据准备

### 方法1：使用POPE数据集（推荐）

POPE数据集专门用于评估目标幻觉问题。

```bash
# 1. 下载COCO val2014图像
wget http://images.cocodataset.org/zips/val2014.zip
unzip val2014.zip -d /path/to/coco/

# 2. 下载POPE问题文件
git clone https://github.com/AoiDragon/POPE.git
cp -r POPE/output D_datasets/POPE/Questions/

# 3. 创建配置文件
cp A_exps/lv1.5_7b_viz.yml A_exps/pope_experiment.yml
```

编辑 `A_exps/pope_experiment.yml`:
```yaml
name_exp: pope_random
name_daset: POPE
name_category: random

path_image_dir: /path/to/coco/val2014
path_question_dir: D_datasets/POPE/Questions
path_model: /path/to/llava-v1.5-7b  # 或 liuhaotian/llava-v1.5-7b
```

### 方法2：使用示例数据集

```bash
# 创建示例数据集
python src/utils/prepare_sample_dataset.py \
    --num_questions 5 \
    --output_dir D_datasets/SAMPLE \
    --create_config

# 添加测试图像到 D_datasets/SAMPLE/Images/
```

---

## 运行实验

### 快速开始

```bash
bash B_scripts/quickstart_visualization.sh
```

### 手动运行

#### 步骤1：运行推理并捕获注意力

```bash
python src/inference_with_visualization.py \
    --device 0 \
    --exp_config A_exps/lv1.5_7b_viz.yml
```

这将：
- 加载LLaVA-1.5-7B模型
- 处理问题并捕获注意力模式
- 保存注意力数据到 `F_visualizations/attention_qid*.pkl`
- 生成预览可视化
- 保存模型响应到 `E_answers/`

#### 步骤2：生成详细可视化

```bash
# 为特定问题生成所有可视化
python src/visualization/visualize_attention.py \
    --attention_file F_visualizations/attention_qid1.pkl \
    --output_dir F_visualizations

# 分析特定层和头
python src/visualization/visualize_attention.py \
    --attention_file F_visualizations/attention_qid1.pkl \
    --layer 15 \
    --head 10 \
    --threshold 0.1
```

---

## 可视化分析

### 生成的可视化类型

1. **注意力热图** (`heatmap_qid*_layer*_head*.png`)
   - 显示完整的注意力矩阵
   - 红框：视觉token区域
   - 橙色虚线框：从视觉token的注意力

2. **汇聚点分析** (`sinks_qid*_layer*_head*.png`)
   - 左图：带有汇聚点标记的热图
   - 右图：视觉token上的注意力分布柱状图
   - 红色标记：识别出的汇聚token

3. **视觉注意力分布** (`vis_attention_dist_qid*_layer*.png`)
   - 显示8个注意力头对视觉token的注意力分布
   - 红色区域：视觉token位置

4. **层间比较** (`layer_comparison_qid*.png`)
   - 比较不同层的注意力模式
   - 显示前4层和后4层

### 可视化示例解读

```
注意力热图示意:
       Key Tokens (键位置) →
Query  ┌─────────────────────────────────┐
(查询  │ 系统 | <image> | 问题 | 回答    │
位置)  │      | (576个) |      |         │
  ↓    │------┼---------┼------┼---------│
       │ 低   │  高     │ 中   │  高     │
       │ 注意力│ 注意力  │注意力│ 注意力   │
       └─────────────────────────────────┘
```

**汇聚点识别：**
- **红色垂直线**：这些是汇聚token
- **高柱状**：接收超过阈值的注意力
- **集中度**：注意力是集中在少数token（汇聚）还是分散

---

## 结果解读

### 预期观察结果

根据论文，你应该观察到：

#### 1. 早期层（0-10）
- 相对分散的注意力
- 汇聚现象不明显

#### 2. 中间层（10-22）
- 清晰的视觉注意力汇聚点出现
- 某些token接收>10%的总注意力
- 不同头的汇聚模式不同

#### 3. 后期层（22-32）
- 持续的汇聚行为
- 可能被VAR机制调节（如果启用）

#### 4. VAR机制的效果

**不使用VAR** (`logic: 0`):
- 注意力强烈集中在少数视觉token上
- 可能导致幻觉（例如看到不存在的物体）

**使用VAR** (`logic: 1`):
- 注意力更均匀分布在视觉token上
- 更好的视觉grounding
- 减少幻觉现象

### 性能对比

在POPE数据集上的预期结果：

| 设置 | Random | Popular | Adversarial |
|------|--------|---------|-------------|
| 不使用VAR | ~87% | ~82% | ~77% |
| 使用VAR | ~89% | ~85% | ~81% |

---

## 参数说明

### VAR机制参数

- **tau (τ)**: 识别汇聚token的阈值
  - 默认值：20
  - 基于RMS norm值识别汇聚候选

- **rho (ρ)**: 允许的最大注意力比例
  - 默认值：0.5
  - 如果对汇聚点的注意力超过此值，触发重分配

- **p**: 注意力重分配因子
  - 默认值：0.6
  - 决定从汇聚点减少多少注意力

- **summ**: 图像的最小总注意力
  - 默认值：0.2
  - 确保足够的注意力保留在视觉内容上

---

## 故障排除

### 问题：看不到明显的汇聚点
- 尝试降低阈值：`--threshold 0.05`
- 检查是否查看了正确的层（尝试15-20层）
- 可视化更多头来找到汇聚行为

### 问题：内存不足
- 减少配置中的 `max_viz_questions`
- 使用更小的批次大小
- 可视化特定层而不是所有层

### 问题：未捕获注意力数据
- 确保配置中 `capture_attention: true`
- 检查使用了 `attn_implementation="eager"`
- 验证生成调用中 `output_attentions=True`

---

## 文档索引

- **[VISUALIZATION_GUIDE.md](VISUALIZATION_GUIDE.md)**: 完整的可视化指南（英文）
- **[EXAMPLE.md](EXAMPLE.md)**: 详细示例演示（英文）
- **[PAPER_NOTES.md](PAPER_NOTES.md)**: 论文实现要点（英文）
- **[README.md](README.md)**: 项目总览

---

## 引用

如果你在研究中使用了本代码，请引用：

```bibtex
@inproceedings{visual-attention-sink,
  title={See What You Are Told: Visual Attention Sink in Large Multimodal Models},
  url={https://arxiv.org/abs/2503.03321},
  year={2024}
}
```

---

## 联系方式

如有问题或建议，请在GitHub仓库中提交issue。

**论文链接**: https://arxiv.org/abs/2503.03321
**项目仓库**: https://github.com/benmagnifico/VisAttnSink
