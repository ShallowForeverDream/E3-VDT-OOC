from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from zipfile import ZipFile, BadZipFile


ROOT = Path(__file__).resolve().parents[1]


REQUIRED_FILES = [
    "README.md",
    "pyproject.toml",
    "requirements.txt",
    "requirements-dev.txt",
    ".github/workflows/ci.yml",
    "demo/app.py",
    "src/e3vdt/inference/vdt_adapter.py",
    "src/e3vdt/inference/pipeline.py",
    "src/e3vdt/inference/cove_attr_pipeline.py",
    "src/e3vdt/attribution/field_nli.py",
    "src/e3vdt/attribution/evidence_relevance.py",
    "src/e3vdt/schemas.py",
    "scripts/check_project.py",
    "scripts/run_cove_attr_demo_cases.py",
    "scripts/run_demo_cases.py",
    "scripts/check_accuracy_preserving.py",
    "scripts/export_demo_outputs.py",
    "scripts/parse_vdt_log.py",
    "scripts/start_demo.ps1",
    "docs/FINAL_DELIVERABLES.md",
    "docs/DEFENSE_QA.md",
    "docs/PRESENTATION_RUNBOOK.md",
    "docs/SYSTEM_DEMO_ACCEPTANCE.md",
    "docs/PROJECT_BRIEF.md",
    "docs/INNOVATION_POINTS.md",
    "docs/ACCURACY_PRESERVING_STRATEGY.md",
    "docs/OUTPUT_SCHEMA.md",
    "docs/DEMO_CASES.md",
    "docs/REPRODUCTION_STATUS.md",
    "docs/report/FINAL_REPORT_DRAFT.md",
    "docs/report/E3-VDT-OOC-结课报告初稿.docx",
    "docs/ppt/PPT_CONTENT_DRAFT.md",
    "docs/ppt/E3-VDT-OOC-答辩PPT初稿.pptx",
    "examples/demo_cases.jsonl",
    "examples/cove_attr_demo_cases.jsonl",
    "examples/cove_attr_demo_outputs.json",
    "examples/demo_outputs.json",
    "examples/reproduction_metrics.json",
]


FORBIDDEN_TRACKED_SUFFIXES = {
    ".pt",
    ".pth",
    ".ckpt",
    ".bin",
    ".safetensors",
    ".npy",
    ".npz",
    ".pkl",
    ".pickle",
    ".tar",
    ".gz",
    ".7z",
    ".rar",
    ".parquet",
    ".sqlite",
    ".db",
}


def fail(message: str, errors: list[str]) -> None:
    errors.append(message)
    print(f"[FAIL] {message}")


def ok(message: str) -> None:
    print(f"[OK] {message}")


def warn(message: str) -> None:
    print(f"[WARN] {message}")


def check_required_files(errors: list[str]) -> None:
    missing = [rel for rel in REQUIRED_FILES if not (ROOT / rel).exists()]
    if missing:
        fail("缺少必要交付文件: " + ", ".join(missing), errors)
    else:
        ok(f"{len(REQUIRED_FILES)} 个必要交付文件均存在")


def check_office_file(rel: str, marker: str, min_slide_count: int | None, errors: list[str]) -> None:
    path = ROOT / rel
    try:
        with ZipFile(path) as zf:
            names = zf.namelist()
            if marker not in names:
                fail(f"{rel} 缺少 Office 结构标记 {marker}", errors)
                return
            if min_slide_count is not None:
                slides = [n for n in names if n.startswith("ppt/slides/slide") and n.endswith(".xml")]
                if len(slides) < min_slide_count:
                    fail(f"{rel} slide 数不足：{len(slides)} < {min_slide_count}", errors)
                    return
        ok(f"{rel} Office zip 结构正常")
    except (BadZipFile, FileNotFoundError) as exc:
        fail(f"{rel} 不是可打开的 Office 文件：{exc}", errors)


def check_demo_cases(errors: list[str]) -> None:
    path = ROOT / "examples/demo_cases.jsonl"
    rows = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            fail(f"demo_cases.jsonl 第 {line_no} 行 JSON 解析失败：{exc}", errors)
            continue
        for key in ["id", "text", "image_context", "expected_label", "expected_mismatch_type", "expected_conflict_fields"]:
            if key not in row:
                fail(f"demo case {row.get('id', line_no)} 缺少字段 {key}", errors)
        rows.append(row)
    if len(rows) < 8:
        fail(f"demo case 数量不足：{len(rows)} < 8", errors)
    else:
        ok(f"demo case 数量充足：{len(rows)}")



def check_cove_attr_demo_cases(errors: list[str]) -> None:
    path = ROOT / "examples" / "cove_attr_demo_cases.jsonl"
    rows = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            fail(f"cove_attr_demo_cases.jsonl 第 {line_no} 行 JSON 解析失败：{exc}", errors)
            continue
        for key in ["sample_id", "current_caption", "true_image_context", "vdt_label", "gold_mismatch_type", "gold_conflict_fields"]:
            if key not in row:
                fail(f"COVE demo case {row.get('sample_id', line_no)} 缺少字段 {key}", errors)
        rows.append(row)
    if len(rows) < 6:
        fail(f"COVE-Attr demo case 数量不足：{len(rows)} < 6", errors)
    else:
        ok(f"COVE-Attr demo case 数量充足：{len(rows)}")
    out = ROOT / "examples" / "cove_attr_demo_outputs.json"
    if out.exists():
        data = json.loads(out.read_text(encoding="utf-8"))
        summary = data.get("summary", {})
        if summary.get("n", 0) < 6 or summary.get("conflict_field_micro_f1", 0.0) < 0.75:
            fail("COVE-Attr demo outputs 自检指标过低或样例数不足", errors)
        else:
            ok("COVE-Attr demo outputs 自检通过")


def check_metrics(errors: list[str]) -> None:
    path = ROOT / "examples/reproduction_metrics.json"
    rows = json.loads(path.read_text(encoding="utf-8"))
    completed = [row for row in rows if row.get("status") == "completed" and row.get("metrics")]
    if not completed:
        fail("reproduction_metrics.json 中没有 completed baseline 指标", errors)
        return
    for row in completed:
        metrics = row["metrics"]
        for key in ["f1", "acc", "auc"]:
            if not isinstance(metrics.get(key), float):
                fail(f"{row['id']} 缺少 float 指标 {key}", errors)
    ok(f"可汇报 completed baseline 数量：{len(completed)}")
    running = [row["id"] for row in rows if row.get("status") == "running_partial"]
    if running:
        warn("仍有 running_partial 实验，最终提交前可再次更新: " + ", ".join(running))


def check_accuracy_preserving_docs(errors: list[str]) -> None:
    targets = [
        ROOT / "src/e3vdt/inference/pipeline.py",
        ROOT / "docs/ACCURACY_PRESERVING_STRATEGY.md",
        ROOT / "docs/OUTPUT_SCHEMA.md",
        ROOT / "docs/FINAL_DELIVERABLES.md",
    ]
    for path in targets:
        text = path.read_text(encoding="utf-8")
        if "baseline_preserving" not in text and "accuracy-preserving" not in text:
            fail(f"{path.relative_to(ROOT)} 未体现 accuracy-preserving / baseline_preserving 约束", errors)
            return
    ok("accuracy-preserving 约束已覆盖代码与关键文档")


def check_tracked_large_or_forbidden(errors: list[str]) -> None:
    try:
        output = subprocess.check_output(["git", "ls-files"], cwd=ROOT, text=True, encoding="utf-8")
    except Exception as exc:  # pragma: no cover - only happens outside git
        warn(f"无法运行 git ls-files，跳过 tracked 大文件检查：{exc}")
        return

    bad: list[str] = []
    huge: list[str] = []
    for rel in output.splitlines():
        path = ROOT / rel
        suffix = path.suffix.lower()
        if suffix in FORBIDDEN_TRACKED_SUFFIXES:
            bad.append(rel)
        if path.exists() and path.is_file() and path.stat().st_size > 25 * 1024 * 1024:
            huge.append(f"{rel} ({path.stat().st_size / 1024 / 1024:.1f} MB)")
    if bad:
        fail("发现禁止提交的大数据/模型文件: " + ", ".join(bad[:10]), errors)
    else:
        ok("未发现已跟踪的大数据/模型后缀文件")
    if huge:
        fail("发现超过 25MB 的已跟踪文件: " + ", ".join(huge[:10]), errors)
    else:
        ok("未发现超过 25MB 的已跟踪文件")


def main() -> int:
    errors: list[str] = []
    print(f"[INFO] checking final deliverables under {ROOT}")
    check_required_files(errors)
    check_office_file("docs/report/E3-VDT-OOC-结课报告初稿.docx", "word/document.xml", None, errors)
    check_office_file("docs/ppt/E3-VDT-OOC-答辩PPT初稿.pptx", "ppt/presentation.xml", 10, errors)
    check_demo_cases(errors)
    check_cove_attr_demo_cases(errors)
    check_metrics(errors)
    check_accuracy_preserving_docs(errors)
    check_tracked_large_or_forbidden(errors)

    if errors:
        print(f"\n[SUMMARY] final deliverable check failed: {len(errors)} error(s)")
        return 1
    print("\n[SUMMARY] final deliverable check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
