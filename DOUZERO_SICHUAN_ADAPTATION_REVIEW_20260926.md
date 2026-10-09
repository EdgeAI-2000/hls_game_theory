# DouZero 到四川血战麻将的迁移分析

日期：2026-09-26。仅分析，未开始实现或训练，不修改运行中的 Mix-PPO。

## 来源与范围

- 论文：Zha et al., DouZero: Mastering DouDizhu with Self-Play Deep Reinforcement Learning, ICML 2021, PMLR 139:12333–12344。
- 论文：https://proceedings.mlr.press/v139/zha21a.html
- PDF：https://proceedings.mlr.press/v139/zha21a/zha21a.pdf
- 官方代码：https://github.com/kwai/DouZero
- 本次阅读 commit：`718a5c920bf3361e34178a38f3b80458e176b351`。
- 本地只读研究副本：`/tmp/douzero_official_review_20260926`；正式复用应另存固定版本并保留 Apache-2.0 LICENSE 和相关版权声明。

## 结论

适合作为独立 RL 方法基线，命名为 **DouZero-Sichuan / DMC-adapted**。
保留 DMC 完整回报回归、状态动作联合编码、LSTM+MLP、epsilon-greedy、并行 actor 与 learner、自博弈。
不使用 Qwen 权重，不复制 Mix-PPO 的搜索、先知、GAE、KL、晋级阈值，也不加入 MASK 的三模块。
合法动作筛选仍遵循项目统一规则。它是算法跨域适配，不是斗地主原环境成绩复现。

## 官方代码对应

| 文件 | 已核对内容 | 四川适配 |
|---|---|---|
| douzero/dmc/models.py | LSTM hidden=128；5个512维隐藏全连接层和1个标量输出层；对候选动作评分 | 保留模型骨架，替换输入维度；输出仍为一个Q值，不是31类softmax |
| douzero/dmc/utils.py:act | 三角色采样、完整终局后回填targets、共享buffer | 四家全部采样，各自标签；等待血战整局结算 |
| douzero/dmc/dmc.py:compute_loss | MSE回归完整回报 | 不改成TD目标、PPO或优势回归 |
| douzero/dmc/utils.py:create_optimizers | RMSprop | 优先保留官方默认参数 |
| douzero/env/env.py | 斗地主合法动作、牌编码、终局ADP/WP/logADP | 环境替换，建议采用线性净分的ADP式目标 |
| douzero/dmc/env_utils.py | 环境包装与终局计数 | 不得把某家先胡当作episode结束 |
| douzero/dmc/arguments.py | 并发、buffer、探索、优化默认值 | 小规模吞吐试验后冻结并发规模；默认total_frames不是论文收敛预算 |

论文的主要优势场景包括斗地主巨大组合动作空间；本项目动作空间较小，不能预设优势必然迁移。

## 环境与自博弈

当前 `mix_ppo_rollout.episode` 仅收集P0轨迹，其他座位是外置策略。
不能只替换P0网络就声称实现DouZero自博弈。新增独立的四座位决策包装器：

1. 复用 `game.py` 的发牌、出牌、响应、终局结算；每个决策返回当前pid、该玩家可见信息和候选动作。
2. 四座位均由学习策略行动并采集数据，初始随机权重加epsilon探索，不以固定规则对手为主训练流程。
3. 优先采用一个参数共享网络，按行动者旋转座位并加入庄家相对位置、响应来源和活跃玩家状态。
4. 地主/农民是非对称角色；四川四家没有这组合作角色，因此共享网络是明确的适配选择，不是原文三个网络的逐字复制。
5. 每局冻结actor参数版本，在局间同步新权重，记录版本差及队列积压；禁止并发加载半更新权重。
6. 已胡玩家不再产生动作，但保留其轨迹，整局结算完后才回填标签。
7. 保留P4当前响应优先级、定缺候选过滤、换三张/定缺初始化等统一约定。不能单独修改DouZero环境规则。

## 可见信息与动作编码

- 牌型计数：27类牌，每类0–4张，用4x27多热表示；自己的手牌属于合法私有信息。
- 公共状态：四家牌河、公开副露、已公开定缺、牌数、是否已胡、分数、牌墙剩余数量、当前阶段、最后弃牌及来源。
- 未见牌：108张全集减去自己手牌及已知公开牌；公开牌必须按物理牌去重，不能把被碰走的弃牌再计一次。
- 禁止输入其他玩家手牌、真实墙顺序、对手oracle向听或等待牌；蒙特卡洛目标可用最终收益，但不能把隐藏信息送入策略输入。
- 官方 `other_hand_cards` 是其他手牌的合并集合；斗地主发完全部牌后该集合可由自身手牌及公共历史推得。麻将有牌墙，必须改成公开未见集合，不能复制读取其他三家真实手牌的做法。
- 历史使用公开事件流，包含相对行动者、动作类型、公开牌；禁止直接编码含其他玩家摸牌内容的完整引擎日志。
- 第一版建议最近32个公开事件、左侧补零并区分padding；该长度是迁移设置，不是官方15步历史的声称复现。
- 动作联合编码：动作类型+涉及牌的计数/牌种+响应上下文。对每个合法候选构造 `(history, state, action)`，批量评分。
- 当前公共接口只有 `d tile/h/g/p/n`，`execute_action` 中自杠取第一个可杠牌。因此第一版沿用这个宏动作接口，并编码其实际指向的牌；不要偷偷给DouZero增加精细选杠能力。若扩展，所有对比方法都要统一接口。

动作选择：`argmax_{a in legal} Q_theta(I,a)`；训练以epsilon概率均匀采样合法动作，评估epsilon=0。

## 血战收益目标

采用终局净分目标，作为显式的ADP式四川适配：

`G_p = (balance_p(final) - balance_p(start)) / C`，第一版建议固定 `C=10`。

每个玩家所有实际选过的 `(I_t,a_t)` 回归自己的 `G_p`；不把三个对手都标为 `-G_P0`。
四人零和只表示四家净分之和为零，并不表示每个对手的收益都等于P0收益的相反数。

本方案将一整局作为终端奖励任务，因此所有时刻使用整局净分。若改为逐笔支付奖励，则必须改用该时刻之后的剩余回报，不能重复混用两套标签。

目标：`mean((Q_theta(I_t,a_t)-G_p)^2)`，gamma=1；无bootstrap、target-Q最大化、GAE和过程塑形奖励。
合法的查大叫、查花猪、杠分和胡牌支付都进入目标，日志按来源分别报告。+7680查大叫必须保留，不能为了好看裁掉。
大额收益可能造成MSE梯度方差；先报告收益分布和梯度裁剪比例，固定线性尺度。若后续改Huber或非线性变换，另命名并披露。

## 建议首版参数与资源

| 项目 | 设置 | 来源/性质 |
|---|---:|---|
| LSTM hidden | 128 | 官方代码 |
| MLP hidden | 5层，每层512 | 官方代码；另有1层标量输出 |
| epsilon | 0.01 | 官方默认 |
| gamma | 1 | 论文完整回报 |
| optimizer | RMSprop，lr=1e-4，alpha=.99，eps=1e-5，momentum=0 | 官方默认 |
| gradient clip | 40 | 官方默认 |
| buffer chunk | 100 transitions | 官方默认；不是每100步截断对局 |
| batch | 32 chunks，即3200 transitions | 官方默认 |
| buffer entries | 50 | 官方默认起点，按输入内存复核 |
| learner | GPU1，1个learner | 本机资源适配 |
| actors | 先4进程，测量后考虑8 | 与现有Mix-PPO共享CPU，不能保证互不影响 |
| actor权重同步 | 局间原子同步 | 可审计实现选择，记录policy version |

无ResNet101和三层搜索，预期每决策更便宜，但不能在吞吐测量前承诺训练时间。
训练预算以真实四家决策数、完成对局、learner更新、墙钟时间分别记录，不能与Mix-PPO的P0决策计数混淆。

## 实现文件建议

- `douzero_sichuan/encoding.py`：观察与动作编码、历史过滤、可见牌去重。
- `douzero_sichuan/env.py`：四座位轮流决策和完整血战episode。
- `douzero_sichuan/model.py`：LSTM+动作条件MLP，合法动作批量Q评分。
- `douzero_sichuan/actors.py`：自博弈、逐玩家轨迹、终局回填、局间同步。
- `douzero_sichuan/train.py`：buffer、RMSprop、恢复随机状态及checkpoint哈希。
- `douzero_sichuan/eval.py`：冻结课程train/holdout、CRN评估、结算分解。
- `douzero_sichuan/tests/`：信息隔离、动作映射、四家收益及终局测试。

不修改运行中的Mix-PPO及其哈希依赖文件，独立目录、tmux、GPU1。

## 先验验证与评估协议

1. 改变隐藏手牌/墙顺序时，给定相同观察和候选动作，编码及Q输出必须不变。
2. 一次胡牌、多次胡牌、杠分、查花猪、查大叫分别与账本对账；无未结算episode进入训练。
3. 检验三位对手标签不是统一取负；先胡玩家的轨迹不会提前丢失；四座位旋转编码一致。
4. 固定episode动作序列比较包装器与当前P4流程，响应顺序、补牌和结算一致。
5. 对一个小固定数据集过拟合，确认Q回归和合法动作排序训练可用；检查所有actor都在产生四家数据。
6. 固定3个训练seed的小预算试验，评估随机/启发式以及冻结train课程；不要只展示自博弈均分，四家对称自博弈平均收益约为零不代表没有学习。
7. 吞吐测量后冻结训练预算和选模/停止标准。只有开发评估可用于选择；正式holdout使用新seeds，所有比较方法同步配对。
8. 主指标为平均净分、荣和对局率、DR、胡牌率与经验CVaR；额外报告结算来源、收益分位数和胡牌机会利用率。
9. excess-DIR若相对MASK no-deceive计算，明确它是跨系统荣和率差，不能据此把DouZero的胜负因果归结为有没有欺骗模块。
10. 没有显式belief输出时DeltaBelief记为不可观测，不能填0后称没有信念变化。神经行为FFR仅作辅助，不混用规则对手的威胁阈值口径。

当前只完成论文、官方代码和项目接口核对。未安装官方依赖、未加载斗地主权重、未启动DouZero训练。
