# Tjong 四川麻将迁移方案

> **执行方案更新：** 本文保留为初步讨论记录，实际实现以 `tjong_sichuan/README.md` 和运行目录的 `PROTOCOL.json` 为准。复核论文第 3.3、4.3、4.4 节后，已补全 3 层 Inner + 3 层因果 Outer、长度 4 的记忆、动作条件化第二次编码和训练期特权 critic。Fan backward 包含非获胜者供牌与放铳的负向归因，不能仅在自己获胜时计算。策略输入依然严格限定为可观察信息。
>
> 首轮冻结预算：8,000 局合成规则数据（7,200 train / 800 validation），最多 20 轮监督训练、patience 4；随后 2,000,000 个四座位自博弈 PPO 决策样本。PPO 训练不是对冻结课程对手训练；课程仅用于监督数据及开发/测试。开发验证每 200,000 次决策执行 1,000 局，选模后使用相同 20×1,000 holdout seeds。GPU1、8 workers、显存分配上限 25%。旧方案中小型单层 Transformer、仅公开 critic、未获胜便无 fan 归因等表述不再适用。

## 论文和代码核查

论文：Li et al., *Tjong: A transformer-based Mahjong AI via hierarchical decision-making and fan backward*，CAAI Transactions on Intelligence Technology 9(4):982–995，2024。

- 论文页面：https://ietresearch.onlinelibrary.wiley.com/doi/full/10.1049/cit2.12298
- 论文 DOI：https://doi.org/10.1049/cit2.12298
- 论文标注的代码仓库：https://github.com/750251985/Tjong

截至 2026-09-28，该 GitHub 地址返回 `404 Repository not found`，无法获得可运行的作者代码。因此本方案属于 **Tjong-adapted**，不能写成作者代码的完整复现。

## 原论文保留的核心思想

Tjong 的核心不是单纯把 MLP 换成 Transformer，而是两个相互配合的设计：

1. **层次化决策**：先决定动作类别，再在该动作类别下选择具体牌或具体操作。
2. **fan backward**：一局结束后，根据最终获胜手的牌型分值，把与最终牌型相关的奖励回溯到本局之前的决策。

论文使用 Transformer/TIT 主干、动作头、吃碰杠等操作选择头和弃牌选择头，并先使用牌局数据监督训练，再使用 PPO 自博弈训练。论文报告约 519,338 条 Botzone 牌局数据、约 15M 参数，并使用约 0.5M 数据进行监督训练。论文还明确区分了 Tjong-1、Tjong-2、Tjong-3 和 Tjong-final，用来分别测试层次决策和 fan backward 的贡献。

## 四川麻将的动作映射

原论文针对中国标准麻将，包含吃、碰、明杠、暗杠、补杠等动作；当前项目是四川血战规则，不能直接照搬动作集合。

| 层级 | 四川麻将适配 |
|---|---|
| Action head | `discard`、`hu`、`gang`、`peng`、`pass` |
| Tile/action head | discard 下选择 27 种牌；gang 下选择当前合法杠牌；没有合法具体选择时只保留对应动作 |
| 缺门约束 | 作为状态特征和合法动作 mask 输入，不能由模型绕过 |
| 合法性 | action head 和 tile head 都使用环境当前合法候选 mask |
| 输出执行 | 先取动作类别，再在该类别候选中取具体牌；避免把 27 张弃牌和所有响应动作混成一个大分类 |

当前实现建议保留 31 个最终宏动作的评测接口，但模型内部采用“动作类别 + 具体选择”两级输出。这样能够和 MASK、DouZero、DRON 共用环境和评测器，同时保留 Tjong 的核心归纳偏置。

## 模型结构

```text
公开局面序列
  ├─ 自己手牌、定缺、牌河、副露、余额、剩余牌数
  ├─ 三个对手的公开动作历史
  └─ 当前响应上下文
          ↓
  Tile/Event Embedding + positional embedding
          ↓
  Transformer encoder
          ↓
  shared latent representation
      ├─ action head: discard / hu / gang / peng / pass
      ├─ discard head: 27 张牌
      └─ response-choice head: 当前合法 gang/peng/hu 选项
          ↓
  hierarchical legal decoding
          ↓
  四川麻将环境动作
```

建议先使用 4–6 层、hidden size 256、4–8 个 attention heads 的小型 Transformer，而不是直接复刻 15M 参数。参数规模必须在协议中冻结；如果使用 15M 结构，应明确这是按论文规模的工程适配，而不是原始 TIT 的代码复现。

输入只能包含 P0 在该时刻能观察到的信息。对手手牌、牌墙真实顺序、oracle 向听数、B_phi 标签和最终胜负牌型不能进入策略输入。最终牌型只能进入训练完成后的 reward shaping。

## Fan backward 的四川适配

当前项目的分值由 `base_score × 2^fan`、自摸、荣和、查叫、花猪等结算项共同决定。原论文的 fan backward 只适合把最终获胜手牌的牌型分数回溯到获胜者轨迹；四川血战还必须处理：

- P0 自摸和荣和的不同责任方；
- 多名玩家先后胡牌；
- 查叫和花猪等终局结算；
- P0 胡牌后仍可能存在其他玩家的终局结算；
- 血战终局中 P0 没有获胜但仍有防守损失的情况。

因此建议定义两个训练奖励：

```text
r_final(t) = P0 在该决策间隔内的实际净分变化 / base_score
r_fan(t)   = 与 P0 最终获胜牌型相关的回溯牌型贡献
r(t)       = r_final(t) + lambda * r_fan(t)
```

`r_fan` 只在训练轨迹回放阶段计算，不能在在线决策时使用终局信息。若 P0 没有获胜，不能凭空构造“获胜牌型”；此时只使用真实净分、放铳责任、查叫和花猪等已结算奖励。第一版建议 `lambda` 预注册为 `0.1`、`0.25` 或 `0.5` 中的一个，并在开发集固定选择，不能根据 holdout 调整。

fan backward 不应替代实际结算奖励，也不应把所有时间步简单赋予同一个最终番数。更合理的做法是根据当前手牌和最终获胜手牌之间的牌计数差异，把奖励分配给仍然保留或促成最终牌型的决策；对放弃最终牌型的动作不给予该项正奖励。

## 训练阶段

### 阶段 A：监督预训练

使用公开合法状态和动作标签训练三个头：

```text
L_BC = L_action + L_discard + L_response
```

数据可以来自已有六个 train 对手和冻结规则策略生成的公开状态，但需要按整局 seed 划分 train/dev，避免同一局相邻状态泄漏。三名 holdout 对手不能参与监督训练、超参数选择或 early stopping。

监督预训练的作用是让层次动作解码先学会合法且基本合理的麻将操作，不能直接被解释为对手泛化或欺骗能力。

### 阶段 B：自博弈/PPO 微调

在预训练 checkpoint 上使用 PPO，策略输出由 action head 和 tile head 联合组成：

```text
log pi(a, tile | s)
  = log pi(action | s)
  + log pi(tile | action, s)
```

价值网络可以共享 Transformer 表征，但输入只包含公开状态。使用经过 fan backward 的训练奖励，同时保留真实净分作为主奖励。

建议首先训练 Tjong-1 和 Tjong-final 两个版本：

| 版本 | 层次决策 | fan backward | 用途 |
|---|---:|---:|---|
| Tjong-1-adapted | 有 | 无 | 检验层次动作结构 |
| Tjong-final-adapted | 有 | 有 | 完整 Tjong 思想 |

如果资源允许，再增加 flat Transformer 和 hierarchical + simple reward，形成与原论文相似的加法消融。

## 正式评测协议

- 训练只使用 6 个 train 课程对手。
- 3 个 holdout 对手只用于最终评测。
- 使用与 P4 相同的 outer seeds：`2026082001`–`2026082020`。
- 每个 outer seed 1,000 局，逐局 seed 为 `outer_seed*1000000+game_index`。
- 固定 checkpoint 后再跑 holdout，不能使用 holdout 选模型。
- 报告平均分、荣和对局率、DR、CVaR-10%、胡牌率和收益来源分解。
- 严格 FFR 作为辅助指标；fan backward 本身不能宣称提高 FFR。

## 与 MASK 的公平边界

Tjong-adapted 不使用 `z`、`B_phi`、GateValue、deceive 候选或 LLM 课程门控。否则它就不再是独立的 Tjong 对比算法。

它与 MASK 的比较应解释为不同策略范式的比较：

- Tjong：Transformer + 层次动作解码 + fan backward；
- MASK：Qwen 基座 + 风险表征 + 选择性动作层塑形 + LLM 课程。

不能把 Tjong 的 fan backward 称为对手信念塑形，也不能预先断言它无法产生诱导荣和。它可能通过更好的牌型规划间接改变对手结果，这正是需要由实验判断的部分。

## 目前结论

论文方法适合迁移，但需要较多规则适配。最大风险不是 Transformer，而是 fan backward 对四川血战多种结算项的语义映射。如果只把最终分数复制给每个动作，实验会变成普通 reward shaping，不能声称实现论文中的 fan backward。

建议先实现离线状态构造、层次动作 mask 和 fan backward 单元测试，再做小规模 PPO smoke；通过后才启动正式训练。当前尚未启动 Tjong 正式训练。
