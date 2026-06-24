from __future__ import annotations
import json, os, socket, sys
from pathlib import Path
try:
    import gradio as gr
except Exception as exc:
    raise SystemExit("Gradio 未安装。请先运行：python -m pip install -r requirements.txt\n" + f"原始错误：{exc}")
ROOT=Path(__file__).resolve().parents[1]; SRC=ROOT/'src'
if str(SRC) not in sys.path: sys.path.insert(0, str(SRC))
from e3vdt.inference.pipeline import E3VDTPipeline
pipe=E3VDTPipeline()

def _load_examples():
    """Load shared demo examples so docs, CLI tests, and Gradio stay aligned."""
    path = ROOT / "examples" / "demo_cases.jsonl"
    rows = []
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            image = None
            if item.get("demo_image"):
                image_path = ROOT / item["demo_image"]
                image = str(image_path) if image_path.exists() else None
            rows.append([image, item["text"], item.get("image_context", "")])
    return rows or [
        [None, "A flood caused evacuations in Shanghai in 2024.", "A flood caused evacuations in Shanghai in 2024."],
        [None, "A large protest erupted in Paris on Monday after a new climate policy.", "People gathered in London during a climate demonstration on Monday."],
        [None, "A fire broke out in New York in 2024.", ""],
    ]

EXAMPLES=_load_examples()


def _load_reproduction_metrics():
    path = ROOT / "examples" / "reproduction_metrics.json"
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def _format_metric_value(value):
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _render_reproduction_metrics():
    rows = _load_reproduction_metrics()
    if not rows:
        return "## 复现实验指标\n\n暂无指标文件。"

    lines = [
        "## VDT strict BLIP-2/GaussianBlur 复现实验指标",
        "",
        "这部分用于答辩展示 baseline 复现状态；大模型权重和原始数据不提交 GitHub，只记录可复现配置与指标。",
        "",
        "| 实验 | 状态 | target_domain 参数 | F1 | Acc | AUC | 备注 |",
        "|---|---|---|---:|---:|---:|---|",
    ]
    for row in rows:
        metrics = row.get("metrics") or {}
        note = row.get("note", "").replace("|", "\\|")
        lines.append(
            "| {id} | {status} | `{target}` | {f1} | {acc} | {auc} | {note} |".format(
                id=row.get("id", "-"),
                status=row.get("status", "-"),
                target=row.get("target_domain_arg", "-"),
                f1=_format_metric_value(metrics.get("f1")),
                acc=_format_metric_value(metrics.get("acc")),
                auc=_format_metric_value(metrics.get("auc")),
                note=note,
            )
        )

    completed = [r for r in rows if r.get("metrics")]
    if completed:
        best = max(completed, key=lambda r: r["metrics"].get("f1", 0.0))
        m = best["metrics"]
        lines += [
            "",
            "### 当前可汇报 baseline",
            "",
            f"- 实验：`{best['id']}`",
            f"- F1：**{m.get('f1', 0):.4f}**",
            f"- Acc：**{m.get('acc', 0):.4f}**",
            f"- AUC：**{m.get('auc', 0):.4f}**",
            "",
            "### 和本项目创新点的关系",
            "",
            "- VDT baseline 负责提供严格论文复现的二分类基线。",
            "- E3-VDT 展示系统进一步输出 `mismatch_type`、`conflict_fields`、`event_scores` 和结构化解释。",
            "- 因此报告里应把 VDT 作为 baseline，把事件字段归因作为系统创新。"
        ]
    return "\n".join(lines)


def run_demo(image, text, image_context):
    image_path=image if isinstance(image,str) else None
    obj=pipe.predict(text=text, image_path=image_path, image_context=image_context).to_dict()
    fields=", ".join(obj["conflict_fields"]) if obj["conflict_fields"] else "无明确冲突 / 证据不足"
    summary=f"### 判断：{obj['label']}\n\n- 置信度：**{obj['confidence']:.2f}**\n- 错配类型：**{obj['mismatch_type']}**\n- 冲突字段：`{fields}`\n- 分类策略：`{obj.get('classification_policy','-')}`\n- 决策来源：`{obj.get('decision_source','-')}`\n\n{obj['explanation']}"
    return summary, obj["event_scores"], json.dumps(obj, ensure_ascii=False, indent=2)


def run_guardrail_demo(text, image_context, baseline_label, baseline_score):
    obj=pipe.predict(
        text=text,
        image_context=image_context,
        baseline_label=baseline_label,
        baseline_score=float(baseline_score),
        classification_policy="baseline_preserving",
    ).to_dict()
    fields=", ".join(obj["conflict_fields"]) if obj["conflict_fields"] else "无明确冲突 / 证据不足"
    preserved = "✅ 已保持 VDT baseline 主分类" if obj["label"] == baseline_label else "❌ 主分类被覆盖，需要检查"
    summary=(
        f"### Accuracy-preserving 验证：{preserved}\n\n"
        f"- VDT baseline label：**{baseline_label}**\n"
        f"- 最终输出 label：**{obj['label']}**\n"
        f"- VDT baseline score：**{obj.get('baseline_score', baseline_score):.2f}**\n"
        f"- 错配类型：**{obj['mismatch_type']}**\n"
        f"- 冲突字段：`{fields}`\n"
        f"- 决策来源：`{obj.get('decision_source','-')}`\n\n"
        "解释模块可以指出冲突字段，但正式策略下不覆盖 VDT 的主分类，因此分类 Accuracy/F1 不会因为 sidecar 模块下降。"
    )
    return summary, obj["event_scores"], json.dumps(obj, ensure_ascii=False, indent=2)


def build_app():
    with gr.Blocks(title="E3-VDT-OOC") as app:
        gr.Markdown("# E3-VDT-OOC 跨域图文内容挪用检测系统\n输入新闻图片和文本，输出 OOC 判断、错配类型、冲突字段和结构化解释。\n\n> 当前 demo 使用轻量可解释 heuristic pipeline 方便展示错配类型；正式实验采用 accuracy-preserving / sidecar 策略：主分类继承 VDT baseline，事件字段模块只做归因，保证分类准确率不低于 baseline。")
        with gr.Tabs():
            with gr.Tab("OOC 检测演示"):
                with gr.Row():
                    with gr.Column(scale=1):
                        image=gr.Image(label="新闻图片（可选；当前 demo 不直接识图）", type="filepath")
                        text=gr.Textbox(label="新闻文本 / Caption", lines=5, placeholder="输入新闻 caption 或 claim")
                        image_context=gr.Textbox(label="图像上下文（建议填写：图像原始 caption / OCR / 检索证据）", lines=5, placeholder="例如：People gathered in London during an earlier climate demonstration in 2020.")
                        btn=gr.Button("开始检测", variant="primary")
                    with gr.Column(scale=1):
                        summary=gr.Markdown(label="结构化判断")
                        scores=gr.Label(label="事件字段一致性分数")
                        raw_json=gr.Code(label="完整 JSON 输出", language="json")
                btn.click(run_demo, inputs=[image,text,image_context], outputs=[summary,scores,raw_json])
                gr.Examples(examples=EXAMPLES, inputs=[image,text,image_context])
                gr.Markdown("## 使用说明\n1. 如果没有图像 caption/OCR，系统会提示证据不足；这是为了避免 demo 假装看懂图片。\n2. 后续接入 BLIP-2/VDT/E3-VDT 后，`image_context` 将由离线缓存或模型自动生成。\n3. 所有模块输出必须遵守 `docs/OUTPUT_SCHEMA.md`。\n4. 样例的期望输出见 `docs/DEMO_CASES.md`，机器可读版本见 `examples/demo_cases.jsonl`。")
            with gr.Tab("分类不降验证"):
                gr.Markdown("## Accuracy-preserving / Sidecar 验证\n这个标签页专门演示：即使事件字段模块发现冲突，最终 `label` 仍然严格继承 VDT baseline。")
                with gr.Row():
                    with gr.Column(scale=1):
                        guard_text=gr.Textbox(label="新闻文本 / Caption", lines=4, value="A large protest erupted in Paris on Monday after a new climate policy.")
                        guard_image_context=gr.Textbox(label="图像上下文 / Evidence", lines=4, value="People gathered in London during a climate demonstration on Monday.")
                        guard_label=gr.Radio(label="VDT baseline label", choices=["OOC","Non-OOC","Uncertain"], value="Non-OOC")
                        guard_score=gr.Slider(label="VDT baseline score", minimum=0.0, maximum=1.0, value=0.91, step=0.01)
                        guard_btn=gr.Button("验证 sidecar 不覆盖主分类", variant="primary")
                    with gr.Column(scale=1):
                        guard_summary=gr.Markdown(label="验证结果")
                        guard_scores=gr.Label(label="事件字段一致性分数")
                        guard_json=gr.Code(label="完整 JSON 输出", language="json")
                guard_btn.click(run_guardrail_demo, inputs=[guard_text,guard_image_context,guard_label,guard_score], outputs=[guard_summary,guard_scores,guard_json])
                gr.Markdown("答辩讲法：第一步说明 VDT 负责二分类；第二步用这个页证明 E3 sidecar 只给归因，不改分类；第三步再切回 OOC 检测演示展示错配类型。")
            with gr.Tab("复现实验指标"):
                gr.Markdown(_render_reproduction_metrics())
    return app


def _pick_port() -> int:
    """Pick a Gradio port that works even when 7860 is already occupied."""
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
