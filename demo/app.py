from __future__ import annotations

import json
import os
import socket
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

try:
    import gradio as gr
except Exception as exc:  # pragma: no cover
    raise SystemExit("Gradio 未安装。请先运行：python -m pip install -r requirements.txt\n" + f"原始错误：{exc}")

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from e3vdt.inference.pipeline import E3VDTPipeline
from e3vdt.inference.cove_attr_pipeline import VDTCOVEAttrPipeline

try:
    from scripts.infer.infer_vdt_cf_attr import predict as predict_vdt_cf_attr
except Exception:
    predict_vdt_cf_attr = None

legacy_pipe = E3VDTPipeline()
cove_pipe = VDTCOVEAttrPipeline()


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _load_legacy_examples():
    rows = []
    for item in read_jsonl(ROOT / "examples" / "demo_cases.jsonl"):
        image = None
        if item.get("demo_image"):
            image_path = ROOT / item["demo_image"]
            image = str(image_path) if image_path.exists() else None
        rows.append([image, item["text"], item.get("image_context", "")])
    return rows or [[None, "A flood caused evacuations in Shanghai in 2024.", "A flood caused evacuations in Shanghai in 2024."]]


LEGACY_EXAMPLES = _load_legacy_examples()
COVE_CASES = read_jsonl(ROOT / "examples" / "cove_attr_demo_cases.jsonl")
CASE_BY_KEY = {f"{r['sample_id']} | {r.get('title','')}": r for r in COVE_CASES}
CASE_KEYS = list(CASE_BY_KEY) or ["manual"]


def _load_no_true_context_examples():
    rows = []
    local_cases = read_jsonl(ROOT / "outputs" / "no_true_context_attr_demo_cases.jsonl")
    if local_cases:
        for item in local_cases:
            p = Path(str(item.get("image", "")))
            if not p.is_absolute():
                p = ROOT / p
            rows.append([
                str(p) if p.exists() else None,
                item.get("caption", ""),
                item.get("vdt_label", "OOC"),
                float(item.get("vdt_score", 0.87) or 0.87),
                item.get("gold_mismatch_type", ""),
            ])
    if rows:
        return rows
    # Fallback examples are only for UI smoke testing.  The no-true-context
    # model may not match these placeholder images; use the exported local
    # VisualNews cases above for a stronger live demo.
    return [
        [str(ROOT / "examples/demo_images/london_climate_demonstration_monday.png"), "A large protest erupted in Paris on Monday after a new climate policy.", "OOC", 0.87, "demo placeholder"],
        [str(ROOT / "examples/demo_images/elon_musk_beijing_2024.png"), "Barack Obama will meet officials in Beijing in 2024.", "OOC", 0.87, "demo placeholder"],
        [str(ROOT / "examples/demo_images/flood_shanghai_2024.png"), "A flood caused evacuations in Shanghai in 2024.", "Non-OOC", 0.91, "demo placeholder"],
    ]


NO_TRUE_CONTEXT_EXAMPLES = _load_no_true_context_examples()


def _default_no_true_context_model() -> str:
    candidates = [
        ROOT / "outputs/no_true_context_attr_5way_1000/no_true_context_attr_head.pkl",
        ROOT / "outputs/no_true_context_attr/no_true_context_attr_head.pkl",
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    return str(candidates[-1])


def _load_reproduction_metrics():
    path = ROOT / "examples" / "reproduction_metrics.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else []


def _fmt(v: Any) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def render_reproduction_metrics() -> str:
    rows = _load_reproduction_metrics()
    if not rows:
        return "## VDT 复现实验\n\n暂无 `examples/reproduction_metrics.json`。"
    lines = [
        "## VDT strict BLIP-2/GaussianBlur 复现实验",
        "",
        "| 实验 | 状态 | target_domain | F1 | Acc | AUC | 说明 |",
        "|---|---|---|---:|---:|---:|---|",
    ]
    for r in rows:
        m = r.get("metrics") or {}
        lines.append(f"| `{r.get('id','-')}` | {r.get('status','-')} | `{r.get('target_domain_arg','-')}` | {_fmt(m.get('f1'))} | {_fmt(m.get('acc'))} | {_fmt(m.get('auc'))} | {r.get('note','').replace('|','/')} |")
    lines += [
        "",
        "**答辩口径**：VDT 是主分类 baseline；COVE-Attr 是旁路归因模块，不覆盖 VDT 的 OOC / Non-OOC 标签。",
    ]
    return "\n".join(lines)


def load_cove_case(key: str):
    row = CASE_BY_KEY.get(key) or (COVE_CASES[0] if COVE_CASES else {})
    return (
        row.get("current_caption", ""),
        row.get("true_image_context", ""),
        row.get("vdt_label", "OOC"),
        float(row.get("vdt_score", 0.87) or 0.87),
        row.get("sample_id", "manual"),
        row.get("image_id", "manual-image"),
        row.get("domain", "demo"),
        row.get("demo_point", ""),
    )


def _field_table(obj: Dict[str, Any]) -> List[List[Any]]:
    rows = []
    for field, item in obj.get("field_nli", {}).items():
        rows.append([
            field,
            item.get("label"),
            ", ".join(item.get("current_values") or []),
            ", ".join(item.get("true_values") or []),
            item.get("confidence"),
            item.get("rationale"),
        ])
    return rows


def _event_table(obj: Dict[str, Any]) -> List[List[Any]]:
    cur, tru = obj.get("current_event", {}), obj.get("true_event", {})
    pairs = [
        ("entity", "entities"),
        ("location", "locations"),
        ("time", "times"),
        ("event_type", "event_types"),
        ("relation", "relations"),
    ]
    return [[name, ", ".join(cur.get(k) or []), ", ".join(tru.get(k) or [])] for name, k in pairs]


def run_cove_attr(case_key: str, current_caption: str, true_context: str, vdt_label: str, vdt_score: float, sample_id: str, image_id: str, domain: str):
    obj = cove_pipe.predict(
        current_caption=current_caption,
        true_image_context=true_context,
        vdt_label=vdt_label,
        vdt_score=float(vdt_score),
        sample_id=sample_id or "manual",
        image_id=image_id or "manual-image",
        domain=domain or "demo",
    )
    conflicts = ", ".join(obj["conflict_fields"]) if obj["conflict_fields"] else "无明确冲突"
    ev = obj["evidence_relevance"]
    summary = f"""
### VDT-COVE-Attr 输出

- **最终主分类**：{obj['final_label']}  
- **主分类来源**：`{obj['decision_source']}`，{obj['classification_note']}  
- **VDT baseline**：{obj['vdt']['label']} / score={obj['vdt']['score']}  
- **错配类型**：**{obj['mismatch_type']}**  
- **冲突字段**：`{conflicts}`  
- **证据相关性**：{ev['level']} / score={ev['score']} / sufficient={ev['sufficient']}  

{obj['explanation']}

> 边界说明：这是系统演示闭环；大规模归因有效性要看人工 gold set + ablation 实验。
""".strip()
    evidence_summary = {
        "relevance_score": ev["score"],
        "level": ev["level"],
        "sufficient": ev["sufficient"],
        "token_overlap": ev["token_overlap"],
        "event_overlap": ev["event_overlap"],
    }
    return summary, evidence_summary, _event_table(obj), _field_table(obj), json.dumps(obj, ensure_ascii=False, indent=2)


def run_no_true_context_attr(image, caption: str, vdt_label: str, vdt_score: float):
    if predict_vdt_cf_attr is None:
        return (
            "### VDT-CF-Attr 未加载\n\n`predict_vdt_cf_attr` 导入失败，请检查 `scripts/infer/infer_vdt_cf_attr.py` 和依赖。",
            {},
            [],
            "{}",
        )
    image_path = image if isinstance(image, str) else ""
    if not image_path:
        return (
            "### 请先上传图片\n\n新版 VDT-CF-Attr 推理阶段不使用 true context，因此必须提供 image + current_caption。",
            {},
            [],
            "{}",
        )
    obj = predict_vdt_cf_attr(
        image_path=image_path,
        caption=caption or "",
        vdt_label=vdt_label or "OOC",
        vdt_score=float(vdt_score),
        model_path=_default_no_true_context_model(),
        device=os.environ.get("VDT_CF_ATTR_DEVICE", "cuda"),
    )
    conflicts = ", ".join(obj.get("conflict_fields") or []) or "无明确冲突"
    summary = f"""
### VDT-CF-Attr 输出（不输入 true context）

- **VDT 主分类**：{obj['vdt_label']} / score={obj['vdt_score']}
- **错配类型**：**{obj['mismatch_type']}**
- **冲突字段**：`{conflicts}`
- **置信度**：{obj['confidence']}
- **视觉证据状态**：{obj['evidence_status']}
- **是否使用 true context**：`{obj['uses_true_context']}`
- **决策来源**：`{obj['decision_source']}`
- **字段存在性后处理**：`{obj.get('postprocess_applied', False)}` / `{obj.get('postprocess_reason', 'no_change')}`

{obj['explanation']}

> 答辩口径：这是最终应用路线的演示入口。它只输入图片和当前 caption；true context 只用于训练构造和评测参考，不进入推理。
""".strip()
    fs = obj.get("feature_summary", {})
    sim_rows = []
    for field, val in (fs.get("prompt_sims") or {}).items():
        sim_rows.append([field, val, (fs.get("field_presence") or {}).get(field, 0)])
    label = {
        "mismatch_type": obj.get("mismatch_type"),
        "confidence": obj.get("confidence"),
        "uses_true_context": obj.get("uses_true_context"),
        "postprocess_applied": obj.get("postprocess_applied"),
        "postprocess_reason": obj.get("postprocess_reason"),
        "clip_caption_sim": fs.get("clip_caption_sim"),
    }
    return summary, label, sim_rows, json.dumps(obj, ensure_ascii=False, indent=2)


def run_legacy_demo(image, text, image_context):
    image_path = image if isinstance(image, str) else None
    obj = legacy_pipe.predict(text=text, image_path=image_path, image_context=image_context).to_dict()
    fields = ", ".join(obj["conflict_fields"]) if obj["conflict_fields"] else "无明确冲突 / 证据不足"
    summary = f"### 判断：{obj['label']}\n\n- 置信度：**{obj['confidence']:.2f}**\n- 错配类型：**{obj['mismatch_type']}**\n- 冲突字段：`{fields}`\n- 决策来源：`{obj.get('decision_source','-')}`\n\n{obj['explanation']}"
    return summary, obj["event_scores"], json.dumps(obj, ensure_ascii=False, indent=2)


def run_guardrail_demo(text, image_context, baseline_label, baseline_score):
    obj = legacy_pipe.predict(
        text=text,
        image_context=image_context,
        baseline_label=baseline_label,
        baseline_score=float(baseline_score),
        classification_policy="baseline_preserving",
    ).to_dict()
    preserved = "✅ 已保持 VDT baseline 主分类" if obj["label"] == baseline_label else "❌ 主分类被覆盖，需要检查"
    fields = ", ".join(obj["conflict_fields"]) if obj["conflict_fields"] else "无明确冲突 / 证据不足"
    summary = (
        f"### Accuracy-preserving 验证：{preserved}\n\n"
        f"- VDT baseline label：**{baseline_label}**\n"
        f"- 最终输出 label：**{obj['label']}**\n"
        f"- 错配类型：**{obj['mismatch_type']}**\n"
        f"- 冲突字段：`{fields}`\n"
        f"- 决策来源：`{obj.get('decision_source','-')}`\n\n"
        "解释模块可以指出冲突字段，但正式策略下不覆盖 VDT 主分类。"
    )
    return summary, obj["event_scores"], json.dumps(obj, ensure_ascii=False, indent=2)


def render_system_dashboard() -> str:
    metric_path = ROOT / "examples" / "cove_attr_demo_outputs.json"
    metric = None
    if metric_path.exists():
        metric = json.loads(metric_path.read_text(encoding="utf-8")).get("summary", {})
    pilot = ""
    if metric:
        pilot = f"""
### 系统演示集自检（curated smoke set）

| 指标 | 数值 |
|---|---:|
| 样例数 | {metric.get('n', 0)} |
| mismatch type acc | {metric.get('mismatch_type_accuracy', 0):.4f} |
| conflict-field micro-F1 | {metric.get('conflict_field_micro_f1', 0):.4f} |
| exact match | {metric.get('exact_match_rate', 0):.4f} |

> 这只是演示集一致性检查，不替代最终人工 gold attribution 实验。
"""
    return f"""
# VDT-COVE-Attr 可验收系统演示

**技术路线**：VDT baseline → controlled counterfactual attribution training → no-true-context image+caption attribution head；COVE-lite true context 仅作为 oracle/构造/评测辅助。

| 模块 | 系统状态 | 答辩口径 |
|---|---|---|
| VDT 主分类 | 已完成两组核心复现 | 提供 OOC / Non-OOC baseline，不声称完整复现全部论文设置 |
| COVE-lite true context | demo 已接入 | 使用 VisualNews 原始 caption/article metadata 作为图片真实语境 |
| Evidence relevance | demo 已接入 | 证据不足时输出 evidence insufficient，避免强行解释 |
| Field-wise NLI attribution | demo 已接入 | 输出 entity/location/time/event_type/relation 字段矛盾 |
| VDT-CF-Attr no-true-context | demo 已接入 | 推理只用 image + caption + VDT score，不输入 true context |
| 大规模归因实验 | 可运行脚本已准备，结果待补 | 答辩后用人工 gold set + ablation 验证解释可靠性 |

{pilot}

**明天演示顺序**：先讲本页路线 → 打开“VDT-CF-Attr 无 true context”跑本地反事实样例 → 打开“VDT-COVE-Attr oracle”说明上限/评测辅助 → 打开“分类不降验证”证明 sidecar 不覆盖 VDT → 打开“复现实验指标”。
""".strip()


def render_experiment_board() -> str:
    lines = [
        "# 实验看板：哪些完成，哪些待补",
        "",
        "| 实验 | 当前状态 | 交付/文件 | 答辩说法 |",
        "|---|---|---|---|",
        "| VDT strict baseline | ✅ 已完成两组核心复现 | `examples/reproduction_metrics.json` | 可汇报分类 baseline 指标 |",
        "| 系统演示集 smoke test | ✅ 已完成 | `examples/cove_attr_demo_outputs.json` | 证明系统闭环和 UI 样例一致 |",
        "| COVE-lite context coverage | ⚠️ 脚本已准备，路径待队友复核 | `scripts/context/build_cove_lite_context_pairs.py` | 答辩后补全 VisualNews metadata 路径后跑 |",
        "| 人工 attribution gold set | ⚠️ 标注表已生成/部分待复核 | `examples/annotation_*_中文.csv` | 需要人工确认，不把弱标签当真 |",
        "| Attribution ablation | ⚠️ 脚本已准备，最终结果待跑 | `scripts/eval/run_attribution_baselines.py` | 比较 majority/random/rule/NLI/evidence relevance |",
        "| NLI/LLM extractor | 规划中 | `vdt_cove_attr_package/` | 作为答辩后增强，不冒充已完成 |",
        "",
        "## 已有本地输出提醒",
    ]
    report = ROOT / "outputs" / "report_tables.md"
    if report.exists():
        txt = report.read_text(encoding="utf-8")
        if "coverage | 0.0" in txt or "| kept | 0 |" in txt:
            lines.append("- 当前 `outputs/report_tables.md` 显示 context coverage 为 0，说明 VisualNews metadata 路径没有对上；不能作为最终实验结果。")
        else:
            lines.append("- `outputs/report_tables.md` 已存在，可用于后续报告表格。")
    else:
        lines.append("- 尚未生成大规模实验 report_tables。")
    return "\n".join(lines)


def build_app():
    with gr.Blocks(title="VDT-COVE-Attr OOC System") as app:
        gr.Markdown(render_system_dashboard())
        with gr.Tabs():
            with gr.Tab("VDT-CF-Attr 无 true context"):
                gr.Markdown("## 最终应用路线：只输入 image + current caption + VDT score\n该页不提供 `true_image_context` 输入；若本地存在 `outputs/no_true_context_attr/no_true_context_attr_head.pkl`，会加载 no-true-context attribution head，否则回退到 field-prompt grounding rule。")
                with gr.Row():
                    with gr.Column(scale=1):
                        ntc_image = gr.Image(label="新闻图片 / Image", type="filepath")
                        ntc_caption = gr.Textbox(label="Current caption / 当前新闻文本", lines=4)
                        ntc_vdt_label = gr.Radio(label="VDT baseline label", choices=["OOC", "Non-OOC", "Uncertain"], value="OOC")
                        ntc_vdt_score = gr.Slider(label="VDT baseline score", minimum=0.0, maximum=1.0, value=0.87, step=0.01)
                        ntc_btn = gr.Button("运行 VDT-CF-Attr", variant="primary")
                    with gr.Column(scale=1):
                        ntc_summary = gr.Markdown()
                        ntc_label = gr.Label(label="No-true-context attribution summary")
                        ntc_sims = gr.Dataframe(headers=["字段", "CLIP prompt similarity", "caption 中是否出现"], label="字段级 prompt grounding 特征", wrap=True)
                        ntc_raw = gr.Code(label="完整 JSON 输出", language="json")
                ntc_btn.click(run_no_true_context_attr, inputs=[ntc_image, ntc_caption, ntc_vdt_label, ntc_vdt_score], outputs=[ntc_summary, ntc_label, ntc_sims, ntc_raw])
                gr.Examples(examples=NO_TRUE_CONTEXT_EXAMPLES, inputs=[ntc_image, ntc_caption, ntc_vdt_label, ntc_vdt_score])

            with gr.Tab("VDT-COVE-Attr 主系统"):
                gr.Markdown("## Oracle/评测辅助：VDT 主分类 + COVE-lite 真实语境 + 字段级归因\n该页会输入 `true_image_context`，因此不要把它说成最终数据集外推理路线。")
                with gr.Row():
                    with gr.Column(scale=1):
                        case = gr.Dropdown(label="选择演示样例", choices=CASE_KEYS, value=CASE_KEYS[0])
                        current = gr.Textbox(label="Current caption / 当前新闻文本", lines=4)
                        true_ctx = gr.Textbox(label="True image context / 图片真实语境", lines=4)
                        vdt_label = gr.Radio(label="VDT baseline label", choices=["OOC", "Non-OOC", "Uncertain"], value="OOC")
                        vdt_score = gr.Slider(label="VDT baseline score", minimum=0.0, maximum=1.0, value=0.87, step=0.01)
                        sample_id = gr.Textbox(label="sample_id", visible=False)
                        image_id = gr.Textbox(label="image_id", visible=False)
                        domain = gr.Textbox(label="domain", visible=False)
                        demo_point = gr.Markdown()
                        run_btn = gr.Button("运行 VDT-COVE-Attr", variant="primary")
                    with gr.Column(scale=1):
                        summary = gr.Markdown()
                        evidence = gr.Label(label="Evidence relevance / sufficiency")
                        event_table = gr.Dataframe(headers=["字段", "current caption", "true image context"], label="事件字段抽取对比", wrap=True)
                        nli_table = gr.Dataframe(headers=["字段", "NLI标签", "当前值", "真实语境值", "置信度", "解释"], label="Field-wise NLI / 字段矛盾判断", wrap=True)
                        raw = gr.Code(label="完整系统 JSON", language="json")
                case.change(load_cove_case, inputs=[case], outputs=[current, true_ctx, vdt_label, vdt_score, sample_id, image_id, domain, demo_point])
                run_btn.click(run_cove_attr, inputs=[case, current, true_ctx, vdt_label, vdt_score, sample_id, image_id, domain], outputs=[summary, evidence, event_table, nli_table, raw])
                app.load(load_cove_case, inputs=[case], outputs=[current, true_ctx, vdt_label, vdt_score, sample_id, image_id, domain, demo_point])

            with gr.Tab("单样本调试/旧版备用"):
                gr.Markdown("该页保留旧版手动 `image_context` 调试能力，用于快速展示字段抽取；主答辩请优先使用第一个 tab。")
                with gr.Row():
                    with gr.Column(scale=1):
                        image = gr.Image(label="新闻图片（可选；当前 demo 不直接识图）", type="filepath")
                        text = gr.Textbox(label="新闻文本 / Caption", lines=5)
                        image_context = gr.Textbox(label="图像上下文 / Evidence", lines=5)
                        btn = gr.Button("开始检测", variant="primary")
                    with gr.Column(scale=1):
                        legacy_summary = gr.Markdown()
                        legacy_scores = gr.Label(label="事件字段一致性分数")
                        legacy_json = gr.Code(label="完整 JSON 输出", language="json")
                btn.click(run_legacy_demo, inputs=[image, text, image_context], outputs=[legacy_summary, legacy_scores, legacy_json])
                gr.Examples(examples=LEGACY_EXAMPLES, inputs=[image, text, image_context])

            with gr.Tab("分类不降验证"):
                gr.Markdown("## Accuracy-preserving / Sidecar 验证\n即使解释模块发现冲突，最终 label 仍继承 VDT baseline。")
                with gr.Row():
                    with gr.Column(scale=1):
                        guard_text = gr.Textbox(label="新闻文本 / Caption", lines=4, value="A large protest erupted in Paris on Monday after a new climate policy.")
                        guard_ctx = gr.Textbox(label="图像上下文 / Evidence", lines=4, value="People gathered in London during a climate demonstration on Monday.")
                        guard_label = gr.Radio(label="VDT baseline label", choices=["OOC", "Non-OOC", "Uncertain"], value="Non-OOC")
                        guard_score = gr.Slider(label="VDT baseline score", minimum=0.0, maximum=1.0, value=0.91, step=0.01)
                        guard_btn = gr.Button("验证 sidecar 不覆盖主分类", variant="primary")
                    with gr.Column(scale=1):
                        guard_summary = gr.Markdown()
                        guard_scores = gr.Label(label="事件字段一致性分数")
                        guard_json = gr.Code(label="完整 JSON 输出", language="json")
                guard_btn.click(run_guardrail_demo, inputs=[guard_text, guard_ctx, guard_label, guard_score], outputs=[guard_summary, guard_scores, guard_json])

            with gr.Tab("复现实验指标"):
                gr.Markdown(render_reproduction_metrics())

            with gr.Tab("实验看板"):
                gr.Markdown(render_experiment_board())
    return app


def _pick_port() -> int:
    explicit = os.environ.get("GRADIO_SERVER_PORT")
    if explicit:
        return int(explicit)
    start = int(os.environ.get("GRADIO_SERVER_PORT_BASE", "7860"))
    host = os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1")
    for port in range(start, start + 30):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind((host, port))
            except OSError:
                continue
            return port
    raise RuntimeError(f"Cannot find empty port in range: {start}-{start + 29}")


if __name__ == "__main__":
    host = os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1")
    port = _pick_port()
    print(f"[E3-VDT-OOC] launching demo: http://{host}:{port}", flush=True)
    build_app().launch(server_name=host, server_port=port)
