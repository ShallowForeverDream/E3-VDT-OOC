# 人工归因标注规范

## 标注目标

给定 current caption 和 true image context，判断它们是否在事件字段上冲突，并标注冲突字段。

## 字段集合

- `entity`：主体、人物、组织、国家、机构。
- `location`：地点、国家、城市、区域、事件发生地。
- `time`：年份、日期、时间段、历史时期。
- `event_type`：事件类别，例如 protest、war/conflict、disaster、politics、court/crime、sports。
- `relation`：主体行为关系，例如 attack、meet、arrest、rescue、win。
- `context_omission`：文本没有显式矛盾，但遗漏关键上下文造成误导。
- `evidence_insufficient`：true context 或 current caption 信息不足，无法可靠判断。

## mismatch_type 取值

- `entity mismatch`
- `location mismatch`
- `temporal mismatch`
- `event-type mismatch`
- `relation mismatch`
- `context omission`
- `uncertain / evidence insufficient`
- `benign illustrative image`

## 标注规则

1. 优先基于 current caption 与 true image context 的文字内容，不使用外部常识补全过多信息。
2. 如果地点明确不同，标 `location`。
3. 如果时间明确不同，标 `time`。
4. 如果主体人物或组织不同，标 `entity`。
5. 如果一个是灾害、一个是政治会议等，标 `event_type`。
6. 如果主体与行为关系冲突，标 `relation`。
7. 如果 true context 太短或没有关键信息，标 `evidence_insufficient`。
8. 可多选 conflict_fields，但 mismatch_type 只选主要冲突类型。

## JSON 模板

```json
{
  "sample_id": "",
  "current_caption": "",
  "true_image_context": "",
  "gold_mismatch_type": "location mismatch",
  "gold_conflict_fields": ["location"],
  "rationale": "current caption and true context disagree on location",
  "annotator": "A",
  "annotation_status": "done"
}
```
