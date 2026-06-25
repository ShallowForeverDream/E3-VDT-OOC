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
  --bg0: #070b16;
  --bg1: #0b1020;
  --card: rgba(255, 255, 255, 0.08);
  --card-strong: rgba(255, 255, 255, 0.13);
  --line: rgba(255, 255, 255, 0.14);
  --text: #edf3ff;
  --muted: #94a3b8;
  --brand: #7dd3fc;
  --brand2: #a78bfa;
  --ok: #34d399;
  --warn: #fbbf24;
  --danger: #fb7185;
}
.gradio-container {
  background:
    radial-gradient(circle at 15% 10%, rgba(125, 211, 252, .20), transparent 32%),
    radial-gradient(circle at 85% 20%, rgba(167, 139, 250, .18), transparent 30%),
    linear-gradient(135deg, var(--bg0), var(--bg1)) !important;
  color: var(--text) !important;
  min-height: 100vh;
}
#main-shell { max-width: 1180px; margin: 0 auto; padding: 52px 28px 64px; }
.hero { margin-bottom: 28px; }
.hero .kicker { color: var(--brand); letter-spacing: .12em; text-transform: uppercase; font-size: 12px; font-weight: 700; }
.hero h1 { font-size: clamp(34px, 5vw, 64px); line-height: 1.04; margin: 8px 0 14px; color: var(--text); }
.hero p { max-width: 760px; color: var(--muted); font-size: 16px; line-height: 1.7; }
.input-panel { background: var(--card); border: 1px solid var(--line); border-radius: 28px; padding: 24px; backdrop-filter: blur(22px); box-shadow: 0 24px 80px rgba(0, 0, 0, .30); }
.input-panel label, .input-panel .label-wrap span { color: var(--text) !important; }
.input-panel textarea, .input-panel input { background: rgba(15, 23, 42, .64) !important; color: var(--text) !important; border-color: var(--line) !important; }
#run-btn { border-radius: 999px !important; min-height: 48px; font-weight: 800; background: linear-gradient(135deg, var(--brand), var(--brand2)) !important; color: #05111f !important; border: 0 !important; }
.result-card { margin-top: 24px; background: linear-gradient(180deg, rgba(255,255,255,.13), rgba(255,255,255,.075)); border: 1px solid var(--line); border-radius: 30px; padding: 28px; color: var(--text); box-shadow: 0 30px 90px rgba(0,0,0,.34); backdrop-filter: blur(26px); }
.card-topline { display: flex; justify-content: space-between; gap: 20px; align-items: flex-start; margin-bottom: 22px; }
.eyebrow { color: var(--brand); letter-spacing: .10em; font-size: 11px; text-transform: uppercase; font-weight: 800; margin: 0 0 8px; }
.result-card h2 { margin: 0; font-size: 26px; color: var(--text); }
.badge { border-radius: 999px; padding: 9px 14px; font-weight: 900; font-size: 13px; border: 1px solid var(--line); }
.badge-ooc { background: rgba(251, 113, 133, .18); color: #fecdd3; }
.badge-ok { background: rgba(52, 211, 153, .16); color: #bbf7d0; }
.badge-warn { background: rgba(251, 191, 36, .16); color: #fde68a; }
.metrics-grid { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 14px; margin: 18px 0 24px; }
.metric-box { background: rgba(15, 23, 42, .58); border: 1px solid var(--line); border-radius: 22px; padding: 18px; min-height: 108px; }
.metric-box.accent { background: linear-gradient(135deg, rgba(125, 211, 252, .18), rgba(167, 139, 250, .18)); }
.metric-label { display: block; color: var(--muted); font-size: 13px; margin-bottom: 10px; }
.metric-box strong { display: block; font-size: 23px; line-height: 1.2; color: var(--text); }
.metric-box small { display: block; color: var(--muted); margin-top: 6px; }
.section-title { font-weight: 900; margin: 0 0 12px; color: var(--text); }
.table-wrap, .explain-box { margin-top: 18px; }
.field-table { width: 100%; border-collapse: collapse; overflow: hidden; border-radius: 18px; background: rgba(15, 23, 42, .48); }
.field-table th, .field-table td { padding: 13px 14px; text-align: left; border-bottom: 1px solid rgba(255,255,255,.10); color: var(--text); }
.field-table th { color: #cbd5e1; font-size: 13px; font-weight: 800; background: rgba(255,255,255,.055); }
.field-table tr:last-child td { border-bottom: 0; }
.pill { display: inline-flex; border-radius: 999px; padding: 5px 10px; background: rgba(148, 163, 184, .16); color: #cbd5e1; font-size: 12px; font-weight: 800; }
.pill-conflict { background: rgba(251, 113, 133, .18); color: #fecdd3; }
.pill-present { background: rgba(125, 211, 252, .16); color: #bae6fd; }
.explain-box { background: rgba(15, 23, 42, .42); border: 1px solid var(--line); border-radius: 22px; padding: 18px; }
.explain-box p { color: #dbeafe; line-height: 1.7; margin: 0 0 12px; }
.meta-line { display: flex; gap: 8px; flex-wrap: wrap; color: var(--muted); font-size: 12px; }
.meta-line span { border: 1px solid var(--line); border-radius: 999px; padding: 6px 10px; }
.raw-json { margin-top: 18px; color: var(--muted); }
.raw-json summary { cursor: pointer; color: #cbd5e1; font-weight: 800; }
.raw-json pre { white-space: pre-wrap; background: rgba(2, 6, 23, .72); color: #dbeafe; padding: 16px; border-radius: 18px; border: 1px solid var(--line); max-height: 360px; overflow: auto; }
.placeholder-card { min-height: 210px; display: flex; flex-direction: column; justify-content: center; }
.placeholder-card p:last-child { color: var(--muted); font-size: 16px; }
.muted { color: var(--muted) !important; }
footer { color: var(--muted); text-align: center; margin-top: 24px; font-size: 12px; }
@media (max-width: 860px) { .metrics-grid { grid-template-columns: repeat(2, 1fr); } .card-topline { flex-direction: column; } }
@media (max-width: 560px) { .metrics-grid { grid-template-columns: 1fr; } #main-shell { padding: 32px 14px 46px; } }
"""


def build_app():
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
