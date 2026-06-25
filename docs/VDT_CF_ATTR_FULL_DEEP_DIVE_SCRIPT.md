# VDT-CF-Attr 全流程深度讲稿（最新版）：基于可控反事实训练的图文错配原因解释

> 更新时间：2026-06-26  
> 面向对象：队友分工、答辩讲解、系统演示前统一口径。  
> 当前仓库：`D:\MY_PROJECT\OOC\E3-VDT-OOC`  
> 当前 GitHub main 最新提交：以仓库 `main` 为准。

---

## 0. 先说结论：原讲稿哪些地方需要修正

原来的 `VDT-CF-Attr-全流程深度讲稿.md` 主方向是对的：

```text
true context 只用于训练构造和评估参考，推理阶段不依赖 true context。
最终系统输入 image + current_caption，先由 VDT 判断是否 OOC，再做错配归因。
```

但原稿有几处已经过时，必须修正后再发给队友：

1. **实验不是“待跑”了**：no-true-context、five-class、plus2000 different-event、真实 OOC 100 条评估都已经有结果。
2. **类别不是只做三类/四类了**：当前稳定版本是五类主标签：
   ```text
   benign illustrative image
   entity mismatch
   location mismatch
   temporal mismatch
   different-event mismatch
   ```
3. **必须写清楚 VDT 的边界**：
   - strict VDT/BLIP-2 baseline 是离线复现实验；
   - 网页 demo 里的 `VDTAdapter` 是 VDT-compatible 自动后端，不等同于完整官方 BLIP-2 VDT checkpoint 在线推理。
4. **必须写清楚系统门控**：只有 VDT 判断为 OOC 时才进入归因；Non-OOC / Uncertain 不强行解释。
5. **必须写清楚队友演示不需要原始数据集**：只跑网页演示不需要 `VisualNews origin.tar`、NewsCLIPpings 和 `configs/paths.local.yaml`，只需要 demo artifact。
6. **必须诚实说明效果**：plus2000 后真实 OOC 100 条指标明显提升，但仍不能说“已经可靠解决真实 OOC 归因”。

下面是修正版讲稿，可以直接发给队友。

---

## 1. 一句话讲清楚我们现在到底做什么

我们的系统是两层结构：

```text
第一层：VDT / VDTAdapter 主分类
输入：image + current_caption
输出：OOC / Non-OOC / Uncertain
回答：图文是否错配？

第二层：VDT-CF-Attr 归因模块
输入：image + current_caption + VDT label/score
输出：mismatch_type + conflict_fields + confidence
回答：如果错配，主要错在哪里？
```

最核心的要求是：

```text
推理阶段不输入 true_image_context。
```

也就是说，最终演示页只让用户上传图片并输入当前新闻文本，不需要用户提供图片的原始新闻上下文。

---

## 2. 系统总流程

### 2.1 最终应用流程

```text
image + current_caption
        ↓
VDT / VDTAdapter 判断 OOC / Non-OOC / Uncertain
        ↓
如果 Non-OOC：输出 benign illustrative image，不进入归因
如果 Uncertain：输出 uncertain / insufficient visual evidence，不强行归因
如果 OOC：进入 VDT-CF-Attr attribution head
        ↓
Attribution Head 基于 image-caption 特征输出错配类型
        ↓
mismatch_type + conflict_fields + confidence + explanation
```

### 2.2 为什么要先 VDT 再归因

因为归因模块回答的是：

```text
如果这对图文已经被判定为错配，那么主要错在哪里？
```

它不是主分类器，不应该覆盖 VDT 的二分类结果。因此当前实现中有明确门控：

| VDT 输出 | 归因模块行为 |
|---|---|
| `OOC` | 进入 attribution head，输出具体错配原因 |
| `Non-OOC` | 直接输出 `benign illustrative image` |
| `Uncertain` | 直接输出 `uncertain / insufficient visual evidence` |

这个设计是为了避免系统在证据不足时胡乱解释。

---

## 3. VDT 与 VDTAdapter 的准确口径

这是答辩和队友演示时最容易被问到的地方，必须统一口径。

### 3.1 我们完成了什么 VDT 复现

我们已经完成 VDT strict BLIP-2/GaussianBlur baseline 的核心离线复现：

| 实验 | 状态 | F1 | Acc | AUC |
|---|---|---:|---:|---:|
| `bbc,guardian`, bs128 | completed | 0.7353 | 0.7383 | 0.7398 |
| `usa_today,washington_post`, bs128 | CUDA OOM | - | - | - |
| `usa_today,washington_post`, bs64 | completed | 0.8032 | 0.8032 | 0.8028 |

这些结果是离线训练/评估日志里的 strict baseline 复现结果。

### 3.2 网页 demo 里为什么不用完整官方 VDT checkpoint

网页 demo 里调用的是：

```text
src/e3vdt/inference/vdt_adapter.py
```

它是一个 **VDT-compatible 自动后端**，作用是让网页不再要求用户手动填写 `VDT label / score`。

它的加载顺序大致是：

```text
1. 如果本机有 no-true-context features，就训练轻量二分类 feature-head
2. 否则尝试 CLIP image-caption similarity fallback
3. 如果是 COVE/oracle 页面且有 image_context，则用事件一致性 fallback
4. 如果证据不足，则输出 Uncertain
```

所以答辩时要这样说：

> 我们 strict 复现了 VDT baseline；系统 demo 中的在线主分类由 VDTAdapter 自动提供 VDT-compatible label/score，用于替代前端手填。当前没有声称网页已经完整接入官方 BLIP-2 VDT checkpoint 在线推理。

这个说法最安全，也最符合当前代码。

---

## 4. 为什么旧版 COVE/true-context 路线要降级为 oracle

旧版路线是：

```text
image + current_caption
  ↓
用 image_id 找 VisualNews true_image_context
  ↓
current_caption vs true_image_context
  ↓
NLI / graph alignment / rule
  ↓
输出 mismatch_type
```

这个思路不是完全错误，它可以作为：

```text
oracle / upper bound / 构造和评测辅助
```

但它不能作为最终应用路线，原因是：

1. 数据集外图片不一定有 VisualNews `image_id -> original caption` 映射；
2. 如果推理阶段依赖 true context，系统就不是“看图和当前文本判断错配原因”，而是“拿当前文本和原始文本比差异”；
3. 真实应用中用户通常只给图片和当前新闻文本，不会给原图原始新闻上下文。

因此新版明确规定：

| 用途 | 是否允许使用 true context |
|---|---:|
| 构造反事实训练样本 | 允许 |
| 构造/分析人工 gold set | 允许作为标注参考 |
| COVE-lite oracle baseline | 允许，但必须单独标注 |
| 最终 no-true-context 推理 | 不允许 |
| 网页最终演示页 | 不允许 |

---

## 5. 训练数据怎么构造

原始 NewsCLIPpings 主要提供的是：

```text
OOC / Non-OOC 二分类标签
```

它没有直接告诉我们：

```text
错的是人物、地点、时间，还是整件事完全不同？
```

所以我们用两类方法构造错配归因标签。

---

## 6. 第一类训练样本：从 Non-OOC 构造单字段反事实

### 6.1 为什么从 Non-OOC 出发

Non-OOC 样本本来是图文匹配的：

```text
image I 与 caption T 匹配
```

如果只替换 caption 里的一个字段，就能得到一个错配原因确定的样本。

例子：

```text
原始 caption：Biden spoke in Washington in 2024.
替换地点：Washington -> Paris
编辑后：Biden spoke in Paris in 2024.
```

此时我们知道错配原因是：

```text
location mismatch
```

### 6.2 当前稳定实现的三种单字段编辑

当前实现主要做三类可控编辑：

| edit_type | gold_mismatch_type | gold_conflict_fields |
|---|---|---|
| `entity_swap` | `entity mismatch` | `entity` |
| `location_swap` | `location mismatch` | `location` |
| `time_swap` | `temporal mismatch` | `time` |

此外保留原始 Non-OOC 作为正样本：

| edit_type | gold_mismatch_type | gold_conflict_fields |
|---|---|---|
| `none` | `benign illustrative image` / `none` | 空 |

### 6.3 字段抽取与替换

当前实现使用：

```text
spaCy NER + 年份正则 + 标题短语规则
```

抽取字段包括：

```text
entity / location / time
```

其中 time 第一版主要做年份替换，因为年份 span 稳定、语法破坏少。

替换不是让大模型重写句子，而是做 span-level 最小编辑：

```text
原句前半部分 + replacement + 原句后半部分
```

这样能最大程度保证只有一个字段变化。

---

## 7. 第二类训练样本：从原始 OOC 筛选 different-event mismatch

### 7.1 为什么要加入原始 OOC

人工标注 100 条真实 OOC 后，我们发现真实 OOC 很多不是单字段错配，而是：

```text
人物、地点、事件类型、关系等多字段一起错
```

也就是更接近：

```text
different-event mismatch
```

如果训练集只有单字段反事实，模型会倾向把真实 OOC 错判成 entity/time/location 单字段错配。

### 7.2 我们如何筛选原始 OOC

不是把所有原始 OOC 都粗暴标成 different-event，而是只选择明显低相似、低重合的 OOC：

```text
NewsCLIPpings similarity_score <= 0.65
caption 与 true_image_context token Jaccard <= 0.08
caption / true_context 至少 4 个 token
```

并且排除人工评测集，避免训练/测试泄漏：

```text
exclude examples/real_ooc_attribution_eval_set.jsonl
按 sample_id / text_id / image_id / source_sample_id 排除
```

本轮统计：

```text
manual_gold_overlap = 94
```

说明有 94 条候选原始 OOC 与人工 gold set 有重合，被排除出训练。

### 7.3 当前 plus2000 训练分布

当前最新版训练集目录：

```text
outputs/no_true_context_attr_5way_plus2000/
```

数据分布为：

| Label | Count |
|---|---:|
| benign illustrative image / none | 1000 |
| entity mismatch | 1000 |
| location mismatch | 1000 |
| temporal mismatch | 1000 |
| different-event mismatch | 3000 |

总计：

```text
7000 条
```

其中 different-event 是在旧五类约 1000 条基础上额外加入约 2000 条原始 OOC。

### 7.4 数据划分与泄漏检查

按 source/image/text 分组划分：

| Split | Rows |
|---|---:|
| train | 4989 |
| val | 1013 |
| test | 998 |

泄漏检查结果：

```text
source_sample_id leakage = 0
image_id leakage = 0
text_id leakage = 0
cross_split duplicate edited_caption = 0
```

这点很重要：同一原始样本的不同反事实版本不能同时出现在 train/test，否则指标会虚高。

---

## 8. no-true-context 特征构造

最终模型推理阶段不能使用 true context，所以特征只来自：

```text
image + current_caption + VDT score
```

当前脚本：

```text
scripts/features/build_image_caption_attribution_features.py
```

### 8.1 当前实际使用的特征

主要包括：

#### 1. 图像加载状态

```text
image_loaded
```

#### 2. VDT 分数

```text
vdt_score
```

当前训练中反事实样本默认给一个 VDT score，真实系统推理时由 VDTAdapter 自动给出。

#### 3. caption 基础特征

```text
caption_chars
caption_tokens
```

#### 4. 字段存在性特征

```text
entity_count / entity_present
location_count / location_present
time_count / time_present
event_type_count / event_type_present
relation_count / relation_present
```

这些特征用于约束模型不要输出 caption 中不存在的字段。

#### 5. CLIP 全局图文相似度

```text
clip_caption_sim
```

表示图片和整句 caption 的整体相似度。

#### 6. 字段级 prompt grounding 相似度

对每个字段构造 prompt：

```text
entity:     a news photo involving {entity}
location:   a news photo taken in {location}
time:       a news photo from {time}
event_type: a news photo about {event_type}
relation:   a news photo showing {relation}
```

然后计算：

```text
clip_prompt_sim_entity
clip_prompt_sim_location
clip_prompt_sim_time
clip_prompt_sim_event_type
clip_prompt_sim_relation
```

以及：

```text
clip_prompt_sim_min_present
clip_prompt_sim_mean_present
clip_prompt_sim_max_present
```

### 8.2 需要注意的限制

CLIP 对地点、时间、具体人物的判断能力有限，尤其是：

```text
图片很难看出年份
普通街景很难判断具体城市
人物脸不清楚时很难确认实体
```

所以系统必须允许：

```text
uncertain / insufficient visual evidence
```

而不是强行解释。

---

## 9. Attribution Head 当前实现

当前训练脚本：

```text
scripts/train/train_no_true_context_attribution_head.py
```

### 9.1 它不是大模型

Attribution Head 是轻量监督分类器。当前实现不是重新训练 VDT，也不是训练 BLIP-2，而是在抽取好的 image-caption 特征上训练分类头。

### 9.2 当前输出

主类型输出：

```text
benign illustrative image
entity mismatch
location mismatch
temporal mismatch
different-event mismatch
```

字段输出支持：

```text
entity
location
time
event_type
relation
```

注意：`different-event mismatch` 默认代表多字段、事件级不一致。代码里对 different-event 的默认字段映射以 `entity/location/event_type` 为主，但真实人工标注中也可能包含 `relation/time`，这也是当前模型仍需改进的地方。

### 9.3 当前比较的模型/消融

训练脚本会比较：

| Method | 作用 |
|---|---|
| `majority` | 多数类 baseline |
| `field_prompt_grounding_rule` | 不训练模型，只用 prompt 相似度规则 |
| `logistic_regression_no_true_context` | 轻量 LR head |
| `mlp_no_clip_prompt` | 去掉 CLIP prompt 特征的消融 |
| `mlp_no_field_presence` | 去掉字段存在性特征的消融 |
| `attr_head_image_caption_mlp` | 当前完整 MLP head |

当前保存的是综合指标最好的模型。

---

## 10. 推理阶段实现细节

当前推理脚本：

```text
scripts/infer/infer_vdt_cf_attr.py
```

默认模型路径：

```text
outputs/no_true_context_attr_5way_plus2000/no_true_context_attr_head.pkl
```

### 10.1 命令行示例

```powershell
python scripts\infer\infer_vdt_cf_attr.py `
  --image outputs\no_true_context_attr_demo_images\1073067__cf_entity_316_0.jpg `
  --caption "Believe it or not Ronnie Bigg thanked the fans of Boston on his way out of Beantown"
```

默认：

```text
--vdt-label auto
```

会先调用 `VDTAdapter` 自动给出 OOC / Non-OOC / Uncertain。

### 10.2 后处理约束

推理时有字段存在性后处理：

```text
如果 caption 中没有 time 字段，就不应该输出 temporal mismatch。
如果 caption 中没有 entity 字段，就不应该输出 entity mismatch。
如果 different-event 只剩一个有效字段，则降级为对应单字段 mismatch。
如果没有有效视觉证据，则输出 uncertain。
```

这个后处理解决了 demo 里“模型说错字段但文本里没有该字段”的尴尬问题。

### 10.3 CLIP 不可用时如何处理

旧版本有一个问题：CLIP 全零时，prompt rule 可能给出接近 1.0 的 benign 置信度。这个已经修复。

现在如果 CLIP/torch/transformers 不可用，系统会输出：

```text
uncertain / insufficient visual evidence
```

而不是假装自己很确定。

---

## 11. 当前实验结果

### 11.1 VDT strict baseline 复现结果

| 实验 | 状态 | F1 | Acc | AUC |
|---|---|---:|---:|---:|
| `bbc,guardian`, bs128 | completed | 0.7353 | 0.7383 | 0.7398 |
| `usa_today,washington_post`, bs128 | CUDA OOM | - | - | - |
| `usa_today,washington_post`, bs64 | completed | 0.8032 | 0.8032 | 0.8028 |

这部分是主分类 baseline 复现，不是归因模块指标。

### 11.2 old no-true-context scaling 结果

在不加入大量 original OOC different-event 的设置下，`MaxPerType=1000` 时：

| Method | Type Acc | Field Micro-F1 | Exact Match |
|---|---:|---:|---:|
| logistic regression no-true-context | 0.5275 | 0.5719 | 0.3250 |

这说明 no-true-context 路线可运行，并且随数据规模增大有提升。

### 11.3 五类 original OOC 初版结果

加入约 987 条 filtered original OOC 作为 `different-event mismatch` 后：

| Method | Type Acc | Field Micro-F1 | Exact Match |
|---|---:|---:|---:|
| logistic_regression_no_true_context | 0.4011 | 0.5841 | 0.3257 |

但 different-event recall 仍较弱，所以我们继续做 plus2000 增强。

### 11.4 plus2000 different-event 增强结果

当前最终训练分布：

```text
none/entity/location/time/different-event = 1000/1000/1000/1000/3000
```

合成 held-out test：

| Method | Type Acc | Field Micro-F1 | Exact Match |
|---|---:|---:|---:|
| majority | 0.4669 | 0.7012 | 0.4669 |
| logistic_regression_no_true_context | 0.3317 | 0.6655 | 0.4228 |
| attr_head_image_caption_mlp | **0.5220** | 0.6876 | 0.3487 |

注意：由于 test 中 different-event 占比高，majority 的部分指标不低。因此这张表不能单独证明真实泛化，只说明 MLP 在类型准确率上超过多数类。

### 11.5 真实 OOC 人工 100 条 no-true-context 评估

这是更关键的评估，因为它看真实 OOC 泛化。

| Model | Type Acc | Field Micro-F1 | Exact Match | 说明 |
|---|---:|---:|---:|---|
| `no_true_context_attr_5way_1000` | 0.0900 | 0.3276 | 0.0300 | different-event 只预测 6 条 |
| `no_true_context_attr_5way_plus2000` | **0.2900** | **0.4781** | 0.0300 | different-event 预测 33 条 |

结论要实事求是：

```text
plus2000 训练分布修正明显提升了真实 OOC 归因表现，
但 Type Acc=0.29 仍不高，说明真实 OOC 归因还没有完全解决。
```

这正好支持我们的研究叙事：

```text
先发现真实 OOC 多为 different-event，
再调整训练分布加入更多原始 OOC，
最后真实 OOC 指标确实改善。
```

---

## 12. 系统演示怎么讲

### 12.1 最终演示入口

网页里最重要的 tab 是：

```text
VDT-CF-Attr 无 true context
```

这个页面只输入：

```text
新闻图片 / Image
Current caption / 当前新闻文本
```

不输入：

```text
true_image_context
VDT label / score
```

系统会自动：

```text
1. 调用 VDTAdapter 得到 OOC / Non-OOC / Uncertain
2. 如果 OOC，加载 no_true_context_attr_head.pkl 做归因
3. 输出 mismatch_type、conflict_fields、confidence、evidence_status
```

### 12.2 当前默认模型加载顺序

```text
outputs/no_true_context_attr_5way_plus2000/no_true_context_attr_head.pkl
→ outputs/no_true_context_attr_5way_1000/no_true_context_attr_head.pkl
→ outputs/no_true_context_attr/no_true_context_attr_head.pkl
→ field-prompt grounding rule fallback
```

### 12.3 队友只跑演示需要什么

队友只跑网页演示时，不需要：

```text
configs/paths.local.yaml
VisualNews 原图 / origin.tar
NewsCLIPpings 原始数据
VDT/BLIP-2 checkpoint
```

只需要：

```text
GitHub 源码
Python 依赖
组长发的 demo artifact zip
```

当前 artifact：

```text
artifacts/e3-vdt-ooc-demo-artifact.zip
```

里面包含：

```text
outputs/no_true_context_attr_5way_plus2000/no_true_context_attr_head.pkl
outputs/no_true_context_attr_5way_plus2000/image_caption_features_*.csv
outputs/no_true_context_attr_demo_cases.jsonl
outputs/no_true_context_attr_demo_images/
少量指标文件
```

队友导入命令：

```powershell
git pull origin main

powershell -ExecutionPolicy Bypass -File .\scripts\import_demo_artifact.ps1 `
  -ZipPath D:\path\to\e3-vdt-ooc-demo-artifact.zip

python -m pip install -r requirements.txt
python -m pip install -e .

powershell -ExecutionPolicy Bypass -File .\scripts\start_demo.ps1 -SkipChecks
```

---

## 13. 项目创新点怎么说，不能过度包装

### 13.1 不要这样说

不要说：

```text
我们提出了一个全新 SOTA OOC 检测模型，超过了 VDT。
```

这不符合事实。我们没有证明主分类超过 VDT。

也不要说：

```text
我们的归因模型已经可靠判断所有真实 OOC 错配原因。
```

真实 OOC 100 条 Type Acc 只有 0.29，不能这样说。

### 13.2 应该这样说

更准确的创新点是：

1. **VDT 后置归因 sidecar**  
   在保留 VDT 主分类结果的基础上，增加一个错配原因解释模块，不覆盖 VDT 的 OOC / Non-OOC 判断。

2. **可控反事实归因训练**  
   原数据集缺少错配类型标签，我们从 Non-OOC 样本做单字段最小编辑，自动构造 entity/location/time 的细粒度监督。

3. **加入真实 OOC 分布修正**  
   人工标注发现真实 OOC 多为 different-event mismatch，因此额外加入 filtered original OOC 作为 different-event 训练样本，并排除人工 gold set 防止泄漏。

4. **no-true-context 推理路线**  
   推理阶段只用 image + current_caption + VDT score，不依赖 VisualNews true context，因此可以用于数据集外样本演示。

5. **证据不足保守输出**  
   如果 VDT 不确定、CLIP 不可用或图片证据不足，系统输出 uncertain，不强行编造解释。

6. **实验证明方向有效但仍有边界**  
   plus2000 后真实 OOC 100 条 Type Acc 从 0.09 提升到 0.29，Field Micro-F1 从 0.328 提升到 0.478，说明训练分布修正有帮助，但仍需进一步提升。

---

## 14. 可能被老师追问的问题与回答

### Q1：你们是不是只是在 VDT 后面加了标签？

回答：

> 不是简单手工加标签。原始 NewsCLIPpings 只有 OOC / Non-OOC 二分类标签，没有错配类型。我们通过可控反事实编辑，从原本匹配的 Non-OOC 样本中自动构造 entity/location/time 的细粒度错配监督；再结合 filtered original OOC 构造 different-event 类。这样 attribution head 可以在不使用 true context 的情况下学习 image-caption 字段冲突模式。

### Q2：你们为什么能知道某条样本是 location mismatch？

回答：

> 对单字段反事实样本，我们知道，因为它是从原始匹配样本出发，只替换了一个地点 span，其余文本保持不变。所以这个合成样本的 gold mismatch type 就是 location mismatch。对于原始 OOC，我们不会武断标所有样本，而是只筛选低相似、低文本重合的 OOC 作为 different-event，并且用人工 100 条评估集检验泛化。

### Q3：原始 OOC 为什么可以标成 different-event？

回答：

> 不是全部原始 OOC 都这样标。我们只筛选 `similarity_score <= 0.65` 且 caption 与 true context token Jaccard <= 0.08 的样本，这些样本更可能是整体事件错配。同时我们排除了人工 gold set 防止泄漏。实验上，加入这些样本后真实 OOC 100 条 Type Acc 从 0.09 提升到 0.29，说明这个训练分布修正是有效的，但我们也承认它还不完美。

### Q4：推理时没有 true context，怎么知道错在哪里？

回答：

> 模型不是直接知道图片原始上下文，而是利用 image-caption 多模态特征：整体 CLIP 相似度、字段级 prompt grounding、字段存在性特征和 VDT score。比如 caption 中出现地点 Paris，但图像与 “a news photo taken in Paris” 的 grounding 弱，而实体/事件类 prompt 相对更匹配，模型可能输出 location mismatch。当然，如果图片本身不足以判断地点或时间，系统会输出 uncertain。

### Q5：你们的真实 OOC 指标不高，怎么说明创新有效？

回答：

> 我们不把结果包装成已经完全解决。我们的贡献是建立了一个可运行、可评测的 VDT 后置归因框架，并证明训练分布修正有明显效果：plus2000 后真实 OOC Type Acc 从 0.09 到 0.29，Field Micro-F1 从 0.328 到 0.478。这个结果说明方向有效，同时也暴露了真实 OOC 归因比合成反事实更难，是后续工作的重点。

### Q6：系统为什么有时候输出 Uncertain？

回答：

> 因为我们不希望系统在证据不足时强行归因。比如 CLIP/torch 没加载、图片无法读取、VDTAdapter 判断不确定，或者 caption 中没有可解释字段，系统就返回 uncertain / insufficient visual evidence。这是保守策略，不是 bug。

### Q7：队友电脑为什么没有 VisualNews 也能跑？

回答：

> 只跑演示不需要原始数据集。训练和抽特征阶段需要 VisualNews/NewsCLIPpings；演示阶段只需要已经导出的 demo artifact，包括训练好的 attribution head、少量 features 和 demo 图片。因此队友用 artifact 就能跑网页，不需要 91GB 原图。

---

## 15. 队友分工建议

### 15.1 复现负责人

负责讲清楚：

```text
VDT strict baseline 离线复现结果
两组 completed 指标
bs128 OOM 的原因和 bs64 替代结果
```

关键文件：

```text
docs/REPRODUCTION_STATUS.md
examples/reproduction_metrics.json
```

### 15.2 系统负责人

负责讲清楚：

```text
网页怎么跑
VDTAdapter 自动分类
OOC gate 后进入 VDT-CF-Attr
demo artifact 怎么导入
```

关键文件：

```text
demo/app.py
scripts/start_demo.ps1
scripts/export_demo_artifact.ps1
scripts/import_demo_artifact.ps1
docs/TEAMMATE_REPRODUCTION.md
```

### 15.3 方法负责人

负责讲清楚：

```text
为什么不用 true context 做最终推理
可控反事实如何构造标签
为什么加入 original OOC different-event
no-true-context 特征有哪些
```

关键文件：

```text
scripts/data/build_controlled_counterfactuals.py
scripts/features/build_image_caption_attribution_features.py
scripts/train/train_no_true_context_attribution_head.py
scripts/infer/infer_vdt_cf_attr.py
```

### 15.4 报告负责人

负责讲清楚：

```text
创新点边界
实验表格
真实 OOC 100 条结果
不足与展望
```

关键文件：

```text
README.md
docs/NO_TRUE_CONTEXT_SCALING_RESULTS.md
outputs/report_tables_v2.md
outputs/real_ooc_no_true_context_eval_metrics.json
```

---

## 16. 最终答辩稿核心段落

可以直接这样讲：

> 我们的系统以 VDT 作为图文是否错配的主分类 baseline。VDT 能判断 OOC / Non-OOC，但不能解释具体错在哪里。因此我们在 VDT 后面设计了 VDT-CF-Attr 归因模块。最开始我们尝试用 VisualNews 的 true context 与当前 caption 比较，这可以作为 oracle，但推理阶段依赖原始上下文，数据集外图片无法保证可用。为此我们改成 no-true-context 路线：训练阶段利用匹配样本构造可控反事实，单独替换实体、地点或时间字段，得到细粒度错配标签；同时根据人工真实 OOC 分析，加入 filtered original OOC 作为 different-event 训练样本。推理阶段，系统只输入 image 和 current_caption，先由 VDTAdapter 自动给出 OOC / Non-OOC / Uncertain；只有 OOC 样本进入 attribution head，输出错配类型和冲突字段。实验上，plus2000 different-event 训练后，真实 OOC 100 条 Type Acc 从 0.09 提升到 0.29，Field Micro-F1 从 0.328 提升到 0.478，说明训练分布修正对真实 OOC 归因有帮助。但我们也诚实说明，真实 OOC 归因仍然困难，当前系统是一个可运行、可评测的归因原型，而不是已经完全解决所有错配解释问题。

---

## 17. 最终一句话总结

```text
VDT-CF-Attr 的核心不是用 true context 直接比文本差异，
而是在训练阶段用可控反事实和筛选原始 OOC 构造错配类型监督，
在推理阶段只依赖 image + current_caption + VDT score，
在 VDT 判断 OOC 后输出可能的错配原因，并在证据不足时保守输出 uncertain。
```

---

## 18. 最需要队友记住的三句话

1. **VDT 负责是否 OOC，VDT-CF-Attr 只负责 OOC 后解释错在哪里。**
2. **最终演示不输入 true context；true context 只用于训练构造、oracle 和评估参考。**
3. **plus2000 different-event 提升了真实 OOC 归因效果，但我们不声称已经完全解决。**
