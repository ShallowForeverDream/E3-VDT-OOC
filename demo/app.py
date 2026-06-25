from __future__ import annotations

import html
import json
import os
import socket
import sys
from pathlib import Path
from typing import Any, Dict, List

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

try:
    from scripts.infer.infer_vdt_cf_attr import predict as predict_vdt_cf_attr
except Exception:  # pragma: no cover
    predict_vdt_cf_attr = None


FIELD_ZH = {
    "entity": "主体 / 人物",
    "location": "地点",
    "time": "时间",
    "event_type": "事件类型",
    "relation": "关系 / 动作",
    "evidence_insufficient": "证据不足",
    "context_omission": "上下文遗漏",
}

TYPE_ZH = {
    "benign illustrative image": "未发现明确错配",
    "none": "未发现明确错配",
    "entity mismatch": "主体 / 人物错配",
    "location mismatch": "地点错配",
    "temporal mismatch": "时间错配",
    "event-type mismatch": "事件类型错配",
    "relation mismatch": "关系 / 动作错配",
    "different-event mismatch": "完全不同事件",
    "uncertain / insufficient visual evidence": "视觉证据不足 / 不确定",
    "uncertain / evidence insufficient": "证据不足 / 不确定",
}


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _default_no_true_context_model() -> str:
    candidates = [
        ROOT / "outputs/no_true_context_attr_5way_plus2000/no_true_context_attr_head.pkl",
        ROOT / "outputs/no_true_context_attr_5way_1000/no_true_context_attr_head.pkl",
        ROOT / "outputs/no_true_context_attr/no_true_context_attr_head.pkl",
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    return str(candidates[-1])


def _pct(x: Any) -> str:
    try:
        return f"{float(x) * 100:.1f}%"
    except Exception:
        return "-"


def _esc(x: Any) -> str:
    return html.escape(str(x if x is not None else ""))


def _status_badge(label: str) -> str:
    low = str(label or "").lower()
    if "non" in low or "benign" in low:
        return "<span class='badge badge-ok'>Non-OOC</span>"
    if "uncertain" in low or "unknown" in low:
        return "<span class='badge badge-warn'>Uncertain</span>"
    return "<span class='badge badge-ooc'>OOC</span>"


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    return rows


def _candidate_image_path(raw: str) -> str | None:
    if not raw:
        return None
    p = Path(str(raw))
    if not p.is_absolute():
        p = ROOT / p
    return str(p) if p.exists() else None


def _load_examples() -> List[List[str]]:
    """Load optional demo examples without returning to the old multi-tab UI."""
    examples: List[List[str]] = []
    for item in _read_jsonl(ROOT / "outputs" / "no_true_context_attr_demo_cases.jsonl"):
        image = _candidate_image_path(str(item.get("image") or item.get("image_path") or item.get("demo_image") or ""))
        caption = str(item.get("caption") or item.get("current_caption") or item.get("text") or "").strip()
        if image and caption:
            examples.append([image, caption])
        if len(examples) >= 8:
            break
    if examples:
        return examples

    fallback = [
        ("examples/demo_images/london_climate_demonstration_monday.png", "A large protest erupted in Paris on Monday after a new climate policy."),
        ("examples/demo_images/elon_musk_beijing_2024.png", "Barack Obama will meet officials in Beijing in 2024."),
        ("examples/demo_images/flood_shanghai_2024.png", "A flood caused evacuations in Shanghai in 2024."),
    ]
    for image_raw, caption in fallback:
        image = _candidate_image_path(image_raw)
        if image:
            examples.append([image, caption])
    return examples


def _field_rows(obj: Dict[str, Any]) -> str:
    fs = obj.get("feature_summary") or {}
    prompt_sims = fs.get("prompt_sims") or {}
    presence = fs.get("field_presence") or {}
    conflict_fields = set(obj.get("conflict_fields") or [])
    rows: List[str] = []
    ordered = ["entity", "location", "time", "event_type", "relation", "evidence_insufficient", "context_omission"]
    for field in ordered:
        if field not in prompt_sims and field not in conflict_fields and field not in presence:
            continue
        zh = FIELD_ZH.get(field, field)
        present = int(presence.get(field, 1 if field in {"evidence_insufficient", "context_omission"} else 0) or 0)
        conflict = field in conflict_fields
        sim_val = prompt_sims.get(field, "")
        try:
            sim = f"{float(sim_val):.4f}"
        except Exception:
            sim = "-"
        status_html = "<span class='pill pill-conflict'>冲突</span>" if conflict else "<span class='pill'>未触发</span>"
        present_html = "<span class='pill pill-present'>存在</span>" if present else "<span class='pill'>未检测</span>"
        rows.append(
            "<tr>"
            f"<td>{_esc(zh)}</td>"
            f"<td>{status_html}</td>"
            f"<td>{present_html}</td>"
            f"<td>{_esc(sim)}</td>"
            "</tr>"
        )
    if not rows:
        rows.append("<tr><td colspan='4' class='muted'>暂无字段级结果</td></tr>")
    return "\n".join(rows)


def _result_card(obj: Dict[str, Any]) -> str:
    mismatch = str(obj.get("mismatch_type") or "uncertain / insufficient visual evidence")
    mismatch_zh = TYPE_ZH.get(mismatch, mismatch)
    vdt_label = str(obj.get("vdt_label") or "Uncertain")
    vdt_score = obj.get("vdt_score", 0.0)
    confidence = obj.get("confidence", 0.0)
    evidence_status = obj.get("evidence_status", "uncertain")
    decision_source = obj.get("decision_source", "-")
    postprocess = obj.get("postprocess_applied", False)
    post_reason = obj.get("postprocess_reason", "no_change")
    explanation = obj.get("explanation", "")
    field_table = _field_rows(obj)
    raw_json = _esc(json.dumps(obj, ensure_ascii=False, indent=2))

    return f"""
    <section class="result-card">
      <div class="card-topline">
        <div>
          <p class="eyebrow">VDT-CF-Attr · No true context inference</p>
          <h2>图文错配检测结果</h2>
        </div>
        {_status_badge(vdt_label)}
      </div>

      <div class="metrics-grid">
        <div class="metric-box">
          <span class="metric-label">OOC 判定</span>
          <strong>{_esc(vdt_label)}</strong>
        </div>
        <div class="metric-box">
          <span class="metric-label">OOC 总可信度</span>
          <strong>{_pct(vdt_score)}</strong>
        </div>
        <div class="metric-box accent">
          <span class="metric-label">错配类型</span>
          <strong>{_esc(mismatch_zh)}</strong>
          <small>{_esc(mismatch)}</small>
        </div>
        <div class="metric-box">
          <span class="metric-label">错配类型分数</span>
          <strong>{_pct(confidence)}</strong>
        </div>
      </div>

      <div class="table-wrap">
        <div class="section-title">冲突字段表</div>
        <table class="field-table">
          <thead>
            <tr><th>字段</th><th>状态</th><th>Caption 中是否存在</th><th>视觉 Prompt 相似度</th></tr>
          </thead>
          <tbody>{field_table}</tbody>
        </table>
      </div>

      <div class="explain-box">
        <div class="section-title">系统说明</div>
        <p>{_esc(explanation)}</p>
        <div class="meta-line">
          <span>uses_true_context=false</span>
          <span>source={_esc(decision_source)}</span>
          <span>evidence={_esc(evidence_status)}</span>
          <span>postprocess={_esc(postprocess)} / {_esc(post_reason)}</span>
        </div>
      </div>

      <details class="raw-json"><summary>查看原始 JSON 输出</summary><pre>{raw_json}</pre></details>
    </section>
    """


def _empty_card(message: str) -> str:
    return f"""
    <section class="result-card placeholder-card">
      <p class="eyebrow">Ready</p>
      <h2>等待输入</h2>
      <p>{_esc(message)}</p>
    </section>
    """


def run_inference(image_path: str | None, caption: str) -> str:
    if predict_vdt_cf_attr is None:
        return _empty_card("后端 infer_vdt_cf_attr.py 导入失败，请检查依赖和仓库路径。")
    if not image_path:
        return _empty_card("请上传一张新闻图片。")
    if not str(caption or "").strip():
        return _empty_card("请输入与图片配对的新闻文本。")
    try:
        obj = predict_vdt_cf_attr(
            image_path=str(image_path),
            caption=str(caption),
            vdt_label="auto",
            model_path=_default_no_true_context_model(),
            device=os.environ.get("VDT_CF_ATTR_DEVICE", "cuda"),
            no_clip=_env_flag("VDT_CF_ATTR_NO_CLIP"),
        )
        return _result_card(obj)
    except Exception as exc:  # pragma: no cover
        return _empty_card(f"推理失败：{exc}")


CUSTOM_CSS = """
:root {
  --page: #f7f8fb;
  --page2: #eef3ff;
  --card: rgba(255, 255, 255, 0.72);
  --card-strong: rgba(255, 255, 255, 0.88);
  --line: rgba(15, 23, 42, 0.10);
  --text: #0f172a;
  --muted: #64748b;
  --brand: #2563eb;
  --brand-soft: #dbeafe;
  --brand2: #7c3aed;
  --ok: #059669;
  --warn: #d97706;
  --danger: #e11d48;
}
.gradio-container {
  background:
    radial-gradient(circle at 8% 8%, rgba(37, 99, 235, .12), transparent 34%),
    radial-gradient(circle at 88% 15%, rgba(124, 58, 237, .10), transparent 30%),
    linear-gradient(135deg, #ffffff 0%, var(--page) 44%, var(--page2) 100%) !important;
  color: var(--text) !important;
  min-height: 100vh;
}
#main-shell { max-width: 1120px; margin: 0 auto; padding: 56px 28px 70px; }
.hero { margin-bottom: 28px; }
.hero .kicker { color: var(--brand); letter-spacing: .12em; text-transform: uppercase; font-size: 12px; font-weight: 800; }
.hero h1 { font-size: clamp(34px, 5vw, 60px); line-height: 1.05; margin: 10px 0 16px; color: var(--text); letter-spacing: -0.04em; }
.hero p { max-width: 760px; color: var(--muted); font-size: 16px; line-height: 1.75; }
.input-panel {
  background: var(--card);
  border: 1px solid rgba(255,255,255,.72);
  border-radius: 30px;
  padding: 26px;
  backdrop-filter: blur(24px);
  box-shadow: 0 24px 70px rgba(15, 23, 42, .10);
}
.input-panel label, .input-panel .label-wrap span { color: var(--text) !important; font-weight: 750 !important; }
.input-panel textarea, .input-panel input {
  background: rgba(255, 255, 255, .82) !important;
  color: var(--text) !important;
  border-color: rgba(15, 23, 42, .12) !important;
  box-shadow: inset 0 1px 0 rgba(255,255,255,.72) !important;
}
#run-btn { border-radius: 999px !important; min-height: 48px; font-weight: 850; background: linear-gradient(135deg, var(--brand), var(--brand2)) !important; color: #fff !important; border: 0 !important; box-shadow: 0 14px 28px rgba(37, 99, 235, .24) !important; }
.examples-title { margin-top: 18px; color: var(--text); font-size: 14px; font-weight: 850; }
.examples-hint { color: var(--muted); font-size: 13px; margin-bottom: 8px; }
.input-panel .examples, .input-panel .gradio-examples { background: rgba(255,255,255,.62) !important; border-radius: 20px !important; border: 1px solid rgba(15,23,42,.08) !important; padding: 10px !important; }
.result-card {
  margin-top: 24px;
  background: linear-gradient(180deg, rgba(255,255,255,.88), rgba(255,255,255,.64));
  border: 1px solid rgba(255,255,255,.82);
  border-radius: 32px;
  padding: 30px;
  color: var(--text);
  box-shadow: 0 30px 90px rgba(15,23,42,.12);
  backdrop-filter: blur(28px);
}
.card-topline { display: flex; justify-content: space-between; gap: 20px; align-items: flex-start; margin-bottom: 22px; }
.eyebrow { color: var(--brand); letter-spacing: .10em; font-size: 11px; text-transform: uppercase; font-weight: 850; margin: 0 0 8px; }
.result-card h2 { margin: 0; font-size: 26px; color: var(--text); letter-spacing: -0.02em; }
.badge { border-radius: 999px; padding: 9px 14px; font-weight: 900; font-size: 13px; border: 1px solid var(--line); background: rgba(255,255,255,.72); }
.badge-ooc { color: var(--danger); background: rgba(225, 29, 72, .08); }
.badge-ok { color: var(--ok); background: rgba(5, 150, 105, .08); }
.badge-warn { color: var(--warn); background: rgba(217, 119, 6, .10); }
.metrics-grid { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 14px; margin: 18px 0 24px; }
.metric-box { background: rgba(255,255,255,.76); border: 1px solid rgba(15,23,42,.08); border-radius: 24px; padding: 18px; min-height: 110px; box-shadow: 0 10px 28px rgba(15,23,42,.05); }
.metric-box.accent { background: linear-gradient(135deg, rgba(219,234,254,.86), rgba(237,233,254,.86)); }
.metric-label { display: block; color: var(--muted); font-size: 13px; margin-bottom: 10px; }
.metric-box strong { display: block; font-size: 23px; line-height: 1.2; color: var(--text); }
.metric-box small { display: block; color: var(--muted); margin-top: 6px; }
.section-title { font-weight: 900; margin: 0 0 12px; color: var(--text); }
.table-wrap, .explain-box { margin-top: 18px; }
.field-table { width: 100%; border-collapse: collapse; overflow: hidden; border-radius: 20px; background: rgba(255,255,255,.74); border: 1px solid rgba(15,23,42,.07); }
.field-table th, .field-table td { padding: 13px 14px; text-align: left; border-bottom: 1px solid rgba(15,23,42,.07); color: var(--text); }
.field-table th { color: #334155; font-size: 13px; font-weight: 850; background: rgba(248,250,252,.86); }
.field-table tr:last-child td { border-bottom: 0; }
.pill { display: inline-flex; border-radius: 999px; padding: 5px 10px; background: rgba(100,116,139,.10); color: #475569; font-size: 12px; font-weight: 850; }
.pill-conflict { background: rgba(225, 29, 72, .10); color: var(--danger); }
.pill-present { background: rgba(37, 99, 235, .10); color: var(--brand); }
.explain-box { background: rgba(255,255,255,.70); border: 1px solid rgba(15,23,42,.08); border-radius: 24px; padding: 18px; }
.explain-box p { color: #334155; line-height: 1.75; margin: 0 0 12px; }
.meta-line { display: flex; gap: 8px; flex-wrap: wrap; color: var(--muted); font-size: 12px; }
.meta-line span { border: 1px solid rgba(15,23,42,.08); border-radius: 999px; padding: 6px 10px; background: rgba(248,250,252,.72); }
.raw-json { margin-top: 18px; color: var(--muted); }
.raw-json summary { cursor: pointer; color: #334155; font-weight: 850; }
.raw-json pre { white-space: pre-wrap; background: rgba(248,250,252,.86); color: #0f172a; padding: 16px; border-radius: 18px; border: 1px solid rgba(15,23,42,.08); max-height: 360px; overflow: auto; }
.placeholder-card { min-height: 210px; display: flex; flex-direction: column; justify-content: center; }
.placeholder-card p:last-child { color: var(--muted); font-size: 16px; }
.muted { color: var(--muted) !important; }
footer { color: var(--muted); text-align: center; margin-top: 24px; font-size: 12px; }
@media (max-width: 860px) { .metrics-grid { grid-template-columns: repeat(2, 1fr); } .card-topline { flex-direction: column; } }
@media (max-width: 560px) { .metrics-grid { grid-template-columns: 1fr; } #main-shell { padding: 34px 14px 46px; } }
"""


def build_app():
    examples = _load_examples()
    with gr.Blocks(title="VDT-CF-Attr") as app:
        with gr.Column(elem_id="main-shell"):
            gr.HTML(
                """
                <div class="hero">
                  <div class="kicker">AI content safety · OOC attribution</div>
                  <h1>图文内容挪用检测与错配归因</h1>
                  <p>上传一张新闻图片并输入当前配文。系统自动进行 OOC 判定，并在不输入 true context 的条件下输出错配类型、可信度与冲突字段。</p>
                </div>
                """
            )
            with gr.Column(elem_classes=["input-panel"]):
                image = gr.Image(label="图片上传", type="filepath", height=320)
                caption = gr.Textbox(
                    label="文字输入",
                    placeholder="输入与图片配对的新闻文本，例如：A large protest erupted in Paris on Monday after a new climate policy.",
                    lines=4,
                )
                run_btn = gr.Button("开始分析", elem_id="run-btn")
                if examples:
                    gr.HTML("<div class='examples-title'>可选示例</div><div class='examples-hint'>点击任一示例即可自动填入图片与新闻文本。</div>")
                    gr.Examples(
                        examples=examples,
                        inputs=[image, caption],
                        label=None,
                        examples_per_page=8,
                    )
            output = gr.HTML(value=_empty_card("上传图片并输入文本后，点击“开始分析”。"))
            gr.HTML("<footer>VDT-CF-Attr · no-true-context image+caption attribution head · true context is not used at inference.</footer>")
            run_btn.click(fn=run_inference, inputs=[image, caption], outputs=output)
    return app


def _pick_port(host: str, preferred: int) -> int:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((host, preferred))
            return preferred
    except OSError:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((host, 0))
            return int(s.getsockname()[1])


if __name__ == "__main__":
    preferred_port = int(os.environ.get("PORT", "7860"))
    host = os.environ.get("HOST", "127.0.0.1")
    port = _pick_port(host, preferred_port)
    build_app().launch(server_name=host, server_port=port, css=CUSTOM_CSS, theme=gr.themes.Base())
