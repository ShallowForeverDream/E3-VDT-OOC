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
            rows.append([None, item["text"], item.get("image_context", "")])
    return rows or [
        [None, "A flood caused evacuations in Shanghai in 2024.", "A flood caused evacuations in Shanghai in 2024."],
        [None, "A large protest erupted in Paris on Monday after a new climate policy.", "People gathered in London during a climate demonstration on Monday."],
        [None, "A fire broke out in New York in 2024.", ""],
    ]

EXAMPLES=_load_examples()
def run_demo(image, text, image_context):
    image_path=image if isinstance(image,str) else None
    obj=pipe.predict(text=text, image_path=image_path, image_context=image_context).to_dict()
    fields=", ".join(obj["conflict_fields"]) if obj["conflict_fields"] else "无明确冲突 / 证据不足"
    summary=f"### 判断：{obj['label']}\n\n- 置信度：**{obj['confidence']:.2f}**\n- 错配类型：**{obj['mismatch_type']}**\n- 冲突字段：`{fields}`\n\n{obj['explanation']}"
    return summary, obj["event_scores"], json.dumps(obj, ensure_ascii=False, indent=2)
def build_app():
    with gr.Blocks(title="E3-VDT-OOC") as app:
        gr.Markdown("# E3-VDT-OOC 跨域图文内容挪用检测系统\n输入新闻图片和文本，输出 OOC 判断、错配类型、冲突字段和结构化解释。\n\n> 当前 demo 使用轻量可解释 heuristic pipeline；VDT/E3-VDT 严格模型训练完成后将替换后端推理模块。")
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
