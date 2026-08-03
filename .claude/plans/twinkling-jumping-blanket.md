# CLARITY-RAG 训练脚本设计（含断点续训）

## 需求

1. 训练过程可被中断（断电、关机等），再次启动时从中断点继续，而非从头开始
2. 训练轮次（epochs）可调整：第一次跑 N 个 epoch，后续可以再跑 M 个 epoch 继续训练

## 设计思路

### 1. 外层循环：epoch 循环 + question 循环

```
for epoch in range(start_epoch, total_epochs):
    for q_idx in range(start_q_idx, len(examples)):
        # 训练一个 question 的 GRPO 步骤
        # 每训练完 N 个 question 保存一次 checkpoint
```

### 2. 检查点（Checkpoint）格式

每个训练任务对应两个文件：

| 文件 | 格式 | 内容 |
|------|------|------|
| `{policy_name}.npz` | NumPy | 策略网络权重（w1, b1, w2, b2） |
| `{policy_name}.ckpt.json` | JSON | 训练进度元数据 |

`ckpt.json` 的结构：

```json
{
  "config": {
    "dataset": "2wikimultihopqa",
    "model": "llama3.2:3b",
    "split": "train",
    "limit": 1000,
    "epochs": 5,
    "group_size": 3,
    "lr": 0.05,
    "max_retrievals": 2,
    "initial_k": 5,
    "k_increment": 5,
    "sample_temperature": 0.7
  },
  "progress": {
    "completed_epochs": 2,
    "current_epoch": 3,
    "current_question_idx": 0,
    "total_questions": 1000
  },
  "metrics": {
    "epoch_returns": [0.544, 0.615],
    "epoch_objectives": [0.012, 0.008],
    "action_counts": {"stop": 320, "retrieve": 210, "reject": 470}
  }
}
```

### 3. 续训逻辑

启动时：

1. 检查 `--output` 路径是否存在 `.ckpt.json` 和 `.npz`
2. 如果存在且 `--resume` 为 True（或自动检测到 checkpoint）：
   - 加载 `.npz` 恢复策略权重
   - 读取 `.ckpt.json` 获取进度
   - 从 `current_epoch` 和 `current_question_idx` 继续
3. 如果不存在，从头开始训练

### 4. 调整训练轮次

用户运行 `python scripts/train_policy.py --epochs 5`，训练了 5 个 epoch。
后来想再加 3 个 epoch：

```
python scripts/train_policy.py --epochs 8 --resume
```

脚本逻辑：
1. 加载 checkpoint，发现 `completed_epochs=5`
2. 比较 `epochs=8 > completed_epochs=5`，继续训练 epoch 6~8
3. 如果 `epochs <= completed_epochs`，打印"已训练完成，无需继续"并退出

### 5. 保存频率

- 每完成一个 epoch 自动保存
- 每完成 N 个 question 也保存（默认 save_every=50，可通过参数调整）
- 使用临时文件 + 重命名策略避免保存过程中断导致文件损坏：
  1. 先保存到 `{name}.tmp.ckpt.json` 和 `{name}.tmp.npz`
  2. 保存成功后重命名为 `{name}.ckpt.json` 和 `{name}.npz`

### 6. 训练循环细节（每个 question）

```
for group_member in range(group_size):
    state = env.reset(question, gold_answer, gold_titles)
    action, old_prob = policy.sample_action(state)
    
    # 执行第一个动作到终止（后续用贪婪策略）
    step_result = env.step(action)
    total_return = step_result.reward
    
    while not step_result.done:
        greedy_action = np.argmax(policy.action_probs(step_result.state))
        step_result = env.step(greedy_action)
        total_return += step_result.reward
    
    group_samples.append({
        "state": state,          # 初始状态
        "action": action,        # 第一个采样动作
        "old_prob": old_prob,    # 该动作的旧概率
        "return": total_return   # 总回报
    })

advantages = group_advantages([s["return"] for s in group_samples])
objective = policy.grpo_update(group_samples, advantages, lr=lr)
```

### 7. CLI 参数

```
--dataset        数据集 (hotpotqa, 2wikimultihopqa, musique)
--split          数据集分片 (默认 train)
--limit          训练样本数限制
--epochs         训练轮次 (默认 3)
--group-size     GRPO 组大小 (默认 3)
--lr             学习率 (默认 0.05)
--max-retrievals 最大检索次数 (默认 2)
--initial-k      初始检索数 (默认 5)
--k-increment    每次增量检索数 (默认 5)
--save-every     每 N 个 question 保存一次 (默认 50)
--seed           随机种子
--index          BM25 索引路径
--output         策略保存路径 (.npz)
--log-out        训练日志保存路径 (.json)
--resume         续训模式 (默认 False)
```

### 8. 文件名约定

自动生成的输出路径格式：

- 策略文件: `artifacts/policies/{dataset}_{model_slug}_policy.npz`
- 检查点文件: `artifacts/policies/{dataset}_{model_slug}_policy.ckpt.json`
- 训练日志: `artifacts/train_logs/{dataset}_{model_slug}_train_log.json`

### 9. 验证方式

```bash
# 第一次训练 3 个 epoch
python scripts/train_policy.py --dataset 2wikimultihopqa --limit 100 --epochs 3

# 中断后续训
python scripts/train_policy.py --dataset 2wikimultihopqa --limit 100 --epochs 3 --resume

# 增加训练轮次到 5
python scripts/train_policy.py --dataset 2wikimultihopqa --limit 100 --epochs 5 --resume
```
