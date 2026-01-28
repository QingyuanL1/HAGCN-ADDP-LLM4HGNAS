# eval.py 使用方法

## 快速开始

### 1. 评估单个最佳架构（推荐）

```bash
python eval.py \
    --model outputs/pdp_20/run_20260124T135054/epoch-49.pt \
    --datasets data/pdp_20.pkl \
    --decode_strategy greedy \
    --val_size 1000 \
    --arch_source best
```

**说明**：自动加载 NAS 搜索出的最佳架构进行评估

---

### 2. 批量评估 Top-5 架构（对比分析）

```bash
python eval.py \
    --model outputs/pdp_20/run_20260124T135054/epoch-49.pt \
    --datasets data/pdp_20.pkl \
    --decode_strategy greedy \
    --val_size 1000 \
    --arch_source history \
    --top_k 5 \
    --save_eval_results
```

**说明**：评估性能最好的 5 个架构，并保存结果到 JSON 文件

**输出示例**：
```
加载了 5 个架构进行评估

============================================================
评估架构: history_rank_1
  Layers: [[2,3,4,2,3,4,2], [3,4,2,3,4,2,4], [4,2,3,4,2,3,2]]
============================================================
Average cost: 55.2341 +- 0.1234

============================================================
评估结果汇总
============================================================

排名 1: history_rank_1
  平均成本: 55.2341
  训练验证成本: 55.2533

排名 2: history_rank_2
  平均成本: 55.5720
  训练验证成本: 55.5720
...

结果已保存到 outputs/pdp_20/run_xxx/eval_results.json
```

---

### 3. 使用默认架构（向后兼容）

```bash
python eval.py \
    --model outputs/pdp_20/old_model/epoch-49.pt \
    --datasets data/pdp_20.pkl \
    --decode_strategy greedy \
    --val_size 1000 \
    --arch_source none
```

**说明**：不使用 NAS 架构，使用默认的全注意力架构

---

## 参数说明

### 必需参数

| 参数 | 说明 | 示例 |
|------|------|------|
| `--model` | 模型文件路径 | `outputs/pdp_20/run_xxx/epoch-49.pt` |
| `--datasets` | 测试数据集路径 | `data/pdp_20.pkl` |
| `--decode_strategy` | 解码策略 | `greedy` / `sampling` / `bs` |

### NAS 架构相关参数（新增）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--arch_source` | `best` | 架构来源：`best`（最佳）/ `history`（Top-K）/ `none`（默认） |
| `--top_k` | `1` | 评估前 K 个架构（配合 `history` 使用） |
| `--save_eval_results` | `False` | 是否保存评估结果到 JSON |

### 其他常用参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--val_size` | `500` | 评估实例数量 |
| `--eval_batch_size` | `512` | 评估批次大小 |
| `--seed` | `8888` | 随机种子 |
| `--no_cuda` | `False` | 禁用 CUDA |

---

## 使用场景

### 场景 1：快速评估最佳架构

**目的**：验证 NAS 搜索出的最佳架构性能

```bash
python eval.py --model outputs/pdp_20/run_20260124T135054/epoch-49.pt --datasets data/pdp_20.pkl --decode_strategy greedy --arch_source best
```

---

### 场景 2：对比多个架构

**目的**：分析不同架构的性能差异

```bash
python eval.py --model outputs/pdp_20/run_20260124T135054/epoch-49.pt --datasets data/pdp_20.pkl --decode_strategy greedy --arch_source history --top_k 10 --save_eval_results
```

**查看结果**：
```bash
cat outputs/pdp_20/run_20260124T135054/eval_results.json
```

---

### 场景 3：大规模评估

**目的**：在更多实例上评估以获得更准确的结果

```bash
python eval.py --model outputs/pdp_20/run_20260124T135054/epoch-49.pt --datasets data/pdp_20.pkl --decode_strategy greedy --val_size 10000 --arch_source best
```

---

### 场景 4：Beam Search 评估

**目的**：使用 Beam Search 获得更好的解

```bash
python eval.py --model outputs/pdp_20/run_20260124T135054/epoch-49.pt --datasets data/pdp_20.pkl --decode_strategy bs --width 10 --arch_source best
```

---

## 常见问题

### Q1: 如何找到可用的模型文件？

```bash
# 查看所有训练好的模型
ls outputs/pdp_20/*/epoch-*.pt

# 查看某个目录的最后一个 epoch
ls outputs/pdp_20/run_20260124T135054/ | grep epoch | sort -V | tail -1
```

### Q2: 如何知道使用了哪个架构？

评估开始时会打印架构信息：
```
评估架构: best
  Layers: [[2,0,1,2,2,0,3], [1,1,4,3,0,4,0], [1,0,2,4,2,3,4]]
  Aggregation: ['mean', 'mean', 'mean']
```

### Q3: 没有 nas_history.json 文件怎么办？

使用 `--arch_source none` 或者程序会自动回退到默认架构：
```bash
python eval.py --model <model_path> --datasets <data_path> --decode_strategy greedy --arch_source none
```

### Q4: 评估结果保存在哪里？

使用 `--save_eval_results` 后，结果保存在：
```
<model_directory>/eval_results.json
```

例如：`outputs/pdp_20/run_20260124T135054/eval_results.json`

---

## 完整示例

### 示例 1：标准评估流程

```bash
# 1. 找到最新的模型
MODEL_DIR="outputs/pdp_20/run_20260124T135054"
LAST_EPOCH=$(ls $MODEL_DIR | grep epoch | sort -V | tail -1)

# 2. 评估最佳架构
python eval.py \
    --model $MODEL_DIR/$LAST_EPOCH \
    --datasets data/pdp_20.pkl \
    --decode_strategy greedy \
    --val_size 1000 \
    --arch_source best \
    --save_eval_results

# 3. 查看结果
cat $MODEL_DIR/eval_results.json
```

### 示例 2：批量对比评估

```bash
# 评估 Top-10 架构并保存
python eval.py \
    --model outputs/pdp_20/run_20260124T135054/epoch-49.pt \
    --datasets data/pdp_20.pkl \
    --decode_strategy greedy \
    --val_size 5000 \
    --arch_source history \
    --top_k 10 \
    --save_eval_results \
    --seed 1234
```

---

## 架构格式说明

### 操作码（0-4）
- **0**: Zero - 屏蔽该关系
- **1**: Attention - 标准多头注意力
- **2**: GCN - 图卷积网络
- **3**: GAT - 图注意力网络
- **4**: MLP - 多层感知机

### 7 种关系
1. **Global**: 全局注意力（所有节点）
2. **P-D_pair**: 取货点到配对送货点
3. **P-P**: 取货点到取货点
4. **P-D_all**: 取货点到所有送货点
5. **D-P_pair**: 送货点到配对取货点
6. **D-D**: 送货点到送货点
7. **D-P_all**: 送货点到所有取货点

### 聚合方式
- **sum**: 求和聚合
- **mean**: 平均聚合
- **max**: 最大值聚合

---

## 测试验证

运行测试脚本验证功能：

```bash
python test_eval_arch.py
```

---

## 更多帮助

查看所有参数：
```bash
python eval.py --help
```

查看详细文档：
```bash
cat eval_arch_usage.md
```
