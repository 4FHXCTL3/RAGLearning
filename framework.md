模块1：状态与不确定性感知 (State & Uncertainty Perception)
- 输入: 用户查询 qq，检索文档 DD，API生成的中间答案 ansans，上一轮文档 Dprev。
- 核心计算 (Semantic Metrics):
  1. 语义熵 (Hsem): 衡量答案与文档的一致性。熵越高，模型越不确定。
  2. 数据不确定性 (Ua): 衡量文档集合内部的冲突程度（文档间相似度方差）。
  3. 认知不确定性 (Ue): 衡量查询与答案的语义距离。距离越远，模型越“不懂”。
  4. 边际信息增益 (ΔI): 衡量新检索文档相对于旧文档的新颖度（余弦距离）。
- 输出状态向量 St:
St=[Hsem,Ua,Ue,ΔI]

---
模块2：GRPO策略智能体 (GRPO Policy Agent)
- 策略网络 (Policy Network):
  - 结构: 极简 MLP (4维输入 → 64维隐藏层 → 3维输出)。
  - 功能: 输入 St，输出3个动作的概率分布 πθ(a∣St)。
- 动作空间 (Action Space) - 简化版:
  - 相比原设计的4个动作，为了适应API环境和快速实验，简化为3个：
    Action 0: Stop (停止并生成): 当 Hsem 低且 Ue 低时，认为答案已收敛。
    Action 1: Retrieve (继续检索): 当 Hsem 高或 Ua 高时，增加 k 值检索更多文档。
    Action 2: Reject (主动拒答): 当 Ue 极高且多次检索无效时，输出“不知道”，避免幻觉。
- GRPO 优化机制 (核心改进):
  - 去 Critic 化: 不再训练 Value Network。
  - Group 采样: 对于当前状态 St，策略网络采样一组动作（例如G=3 个）：{a1,a2,a3}。
  - 环境执行: 将这3个动作发送给 Module 3执行，获得3个奖励{R1,R2,R3}。
  - 相对优势计算: 利用组内奖励的均值和标准差计算优势函数 (Advantage)：
  Ai=std(Rgroup)Ri−mean(Rgroup)
  - 梯度更新: 使用 Clip 目标函数更新 θ，最大化高优势动作的概率。

---
模块3：环境与奖励优化 (Environment & Reward Optimization)
- 执行者 (Generator):
  - 调用 SiliconFlow API (Qwen2.5-7B)。
  - 根据动作执行：如果是 Retrieve，则增加检索数量；如果是 Stop，则生成最终答案。
  
- 奖励计算 (Composite Reward Function):
  - 采用固定权重的复合奖励：
    R=α⋅Racc−β⋅Rcost+γ⋅Rhonesty+λ⋅ΔI
  
  - Racc (准确性): 最终答案与 Ground Truth 的 EM/F1 匹配度（仅在 Stop 时计算）。
  
  - Rcost (成本惩罚): 每次 API 调用给予负奖励，鼓励少次检索。

  - Rhonesty (诚实度): 正确拒答（确实不知道的问题）给予正奖励。
  
  - ΔI (信息增益): 鼓励检索到与之前文档差异大的新信息。
  
    

![img](https://my.feishu.cn/space/api/box/stream/download/asynccode/?code=M2FlODg2NmRkMDAyZDZjY2U0M2MxYmYyZmNmYWU5ZTZfME56V29neTVZeDhYb3hWN0JQWnA2YmJRTUlqOVFQdDhfVG9rZW46RzB4cmJ0VHhtbzhZUDJ4dmhMZmMxaDRZbjNnXzE3ODExNDQyODA6MTc4MTE0Nzg4MF9WNA&add_watermark=true&scene_type=CCM)
