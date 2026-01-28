# eval.py 架构评估功能使用说明

## 概述

`eval.py` 已经升级，现在支持评估通过 LLM4HGNAS 搜索出来的动态架构。主要功能包括：

- ✅ 从 `nas_history.json` 自动加载最佳架构
- ✅ 支持批量评估 Top-K 架构
- ✅ 支持新旧两种架构格式
- ✅ 完全向后兼容（无架构文件时使用默认架构）
- ✅ 可选保存评估结果到 JSON 文件

## 新增命令行参数

### `--arch_source`
指定架构来源，可选值：
- `best`（默认）：使用 `nas_history.json` 中的最佳架构
- `history`：从历史记录中选择 Top-K 架构
- `none`：使用默认架构（向后兼容模式）

### `--top_k`
指定评估前 K 个架构（配合 `--arch_source history` 使用）
- 默认值：1
- 示例：`--top_k 5` 评估性能最好的 5 个架构

### `--save_eval_results`
将评估结果保存到 JSON 文件
- 保存位置：`<model_dir>/eval_results.json`
- 包含：每个架构的平均成本、标准差、评估时间等

## 使用示例

### 1. 评估最佳架构（推荐）

```bash
python eval.py \
    --model outputs/pdp_20/test_agg_search_20260126T142547/epoch-49.pt \
    --datasets data/pdp_20.pkl \
    --decode_strategy greedy \
    --val_size 1000
```

**说明**：默认使用 `--arch_source best`，自动加载 `nas_history.json` 中的最佳架构。

### 2. 批量评估 Top-5 架构

```bash
python eval.py \
    --model outputs/pdp_20/run_20260124T135054/epoch-49.pt \
    --datasets data/pdp_20.pkl \
    --decode_strategy greedy \
    --arch_source history \
    --top_k 5 \
    --save_eval_results
```

**输出示例**：
```
加载了 5 个架构进行评估

============================================================
评估架构: history_rank_1
  Layers: [[2,3,4,2,3,4,2], [3,4,2,3,4,2,4], [4,2,3,4,2,3,2]]
============================================================
Average cost: 55.2341 +- 0.1234
...

============================================================
评估结果汇总
============================================================

排名 1: history_rank_1
  平均成本: 55.2341
  标准差: 2.1456
  训练验证成本: 55.2533

排名 2: history_rank_2
  平均成本: 55.5720
  标准差: 2.0123
  训练验证成本: 55.5720
...

结果已保存到 outputs/pdp_20/run_xxx/eval_results.json
```

### 3. 使用默认架构（向后兼容）

```bash
python eval.py \
    --model outputs/pdp_20/old_model/epoch-49.pt \
    --datasets data/pdp_20.pkl \
    --decode_strategy greedy \
    --arch_source none
```

**说明**：适用于没有 NAS 搜索的旧模型，使用全注意力默认架构。

### 4. 完整参数示例

```bash
python eval.py \
    --model outputs/pdp_20/test_agg_search_20260126T142547/epoch-49.pt \
    --datasets data/pdp_20.pkl \
    --decode_strategy greedy \
    --val_size 1000 \
    --eval_batch_size 512 \
    --arch_source best \
    --save_eval_results \
    --seed 1234
```

## 架构格式说明

### 新格式（带聚合方式）
```json
{
  "layers": [
    [4, 0, 1, 0, 3, 0, 0],  // 第1层：7个关系的操作码
    [1, 4, 0, 0, 1, 3, 0],  // 第2层
    [1, 2, 2, 1, 3, 2, 3]   // 第3层
  ],
  "aggregation": ["mean", "mean", "sum"]  // 每层的聚合方式
}
```

### 旧格式（简单列表）
```json
[
  [2, 3, 4, 2, 3, 4, 2],  // 第1层
  [3, 4, 2, 3, 4, 2, 4],  // 第2层
  [4, 2, 3, 4, 2, 3, 2]   // 第3层
]
```

**操作码说明**：
- 0: Zero（屏蔽该关系）
- 1: Attention（标准注意力）
- 2: GCN（图卷积）
- 3: GAT（图注意力）
- 4: MLP（线性变换）

**7种关系**：
1. Global（全局注意力）
2. P-D_pair（取货到配对送货）
3. P-P（取货到取货）
4. P-D_all（取货到所有送货）
5. D-P_pair（送货到配对取货）
6. D-D（送货到送货）
7. D-P_all（送货到所有取货）

## 评估结果文件格式

保存的 `eval_results.json` 格式：

```json
{
  "model": "outputs/pdp_20/run_xxx/epoch-49.pt",
  "dataset": "data/pdp_20.pkl",
  "width": 0,
  "timestamp": "2026-01-28 12:34:56",
  "results": {
    "best": {
      "arch": {...},
      "avg_cost": 55.2341,
      "std_cost": 2.1456,
      "min_cost": 51.2345,
      "max_cost": 59.3456,
      "eval_time": 45.32,
      "train_val_cost": 55.2533
    }
  }
}
```

## 常见问题

### Q1: 如果没有 nas_history.json 文件会怎样？
**A**: 程序会显示警告并自动使用默认架构（全注意力），不会报错。

### Q2: 如何知道使用了哪个架构？
**A**: 评估开始时会打印架构信息，包括每层的操作码和聚合方式。

### Q3: 可以同时评估多个模型吗？
**A**: 目前不支持，需要分别运行多次。但可以使用 `--arch_source history --top_k N` 评估同一模型的多个架构。

### Q4: 评估结果与训练时的 val_cost 不一致？
**A**: 这是正常的，因为：
- 训练时使用的是验证集
- 评估时可能使用不同的数据集
- 解码策略可能不同（greedy vs sampling）

## 测试验证

运行测试脚本验证功能：

```bash
python test_eval_arch.py
```

测试内容包括：
- ✅ 加载最佳架构
- ✅ 加载 Top-K 架构
- ✅ 向后兼容性（无架构文件）
- ✅ 架构格式验证（新旧格式）

## 技术细节

### 架构加载流程
1. 从模型目录读取 `nas_history.json`
2. 根据 `arch_source` 选择架构（best/history）
3. 验证架构格式和有效性
4. 传递给 `AttentionModel` 初始化
5. 调用 `model.set_arch()` 设置架构

### 多进程支持
修改后的代码完全支持多进程评估（`--multiprocessing`），架构参数会正确传递给每个进程。

### 向后兼容性
- 如果 `nas_history.json` 不存在，自动使用 `arch=None`
- `arch=None` 时模型使用默认的全注意力架构
- 旧代码无需修改即可继续使用

## 相关文件

- `eval.py`: 主评估脚本（已修改）
- `test_eval_arch.py`: 测试脚本
- `nas_history.json`: NAS 搜索历史（由 `run.py` 生成）
- `eval_results.json`: 评估结果（可选生成）

## 更新日志

**2026-01-28**
- ✅ 添加架构加载功能
- ✅ 支持批量评估 Top-K 架构
- ✅ 添加架构格式验证
- ✅ 支持新旧两种架构格式
- ✅ 添加评估结果保存功能
- ✅ 完全向后兼容
