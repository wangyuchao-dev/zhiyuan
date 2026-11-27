"""两个版本的对比：错误修复率与回归。

为什么单独算这个：
    TSR 涨 3 个点可能是运气（随机采样过了几条）。按 case_id 对齐后能回答更关键的问题：
      · 上一版错的，这一版改对了吗？（错误修复率）
      · 上一版对的，这一版弄坏了吗？（回归）
    前者说明改动有效，后者说明改动有代价。两个都要看。

用法：
    python eval/compare.py 旧结果.json 新结果.json
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any


def load(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def first_run(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """每个用例只取第 1 次运行 —— 与 TSR 的口径保持一致。"""
    out: dict[str, dict[str, Any]] = {}
    for r in payload.get("records", []):
        if int(r.get("run", 0)) != 0:
            continue
        out.setdefault(r["case_id"], r)
    return out


def compare(prev: dict[str, Any], curr: dict[str, Any]) -> dict[str, Any]:
    a, b = first_run(prev), first_run(curr)
    common = sorted(set(a) & set(b))

    prev_fail = [c for c in common if not a[c]["ok"]]
    prev_pass = [c for c in common if a[c]["ok"]]
    fixed = [c for c in prev_fail if b[c]["ok"]]
    still = [c for c in prev_fail if not b[c]["ok"]]
    broke = [c for c in prev_pass if not b[c]["ok"]]

    return {
        "cases": len(common),
        "prev_tsr": len(prev_pass) / len(common) if common else 0.0,
        "curr_tsr": sum(1 for c in common if b[c]["ok"]) / len(common) if common else 0.0,
        "prev_failed": len(prev_fail),
        "fixed": fixed,
        "still_failed": still,
        "regressions": broke,
        # 分母为 0 时返回 None，避免用 0.0 假装"一个都没修好"
        "repair_rate": (len(fixed) / len(prev_fail)) if prev_fail else None,
        "regression_rate": (len(broke) / len(prev_pass)) if prev_pass else None,
        "only_in_prev": sorted(set(a) - set(b)),
        "only_in_curr": sorted(set(b) - set(a)),
    }


def to_markdown(c: dict[str, Any], prev_name: str = "prev", curr_name: str = "curr") -> str:
    def pct(x: float | None) -> str:
        return "—" if x is None else f"{x * 100:.1f}%"

    lines = [
        f"### {curr_name} 相对 {prev_name}",
        "",
        f"- 可比用例：{c['cases']} 条（只出现的用例不计入，见文末）",
        f"- TSR：{pct(c['prev_tsr'])} → {pct(c['curr_tsr'])}",
        f"- **错误修复率**：{pct(c['repair_rate'])}"
        f"（上版失败 {c['prev_failed']} 条，修好 {len(c['fixed'])} 条）",
        f"- **回归率**：{pct(c['regression_rate'])}（上版通过但这次挂了 {len(c['regressions'])} 条）",
        "",
    ]
    if c["fixed"]:
        lines.append(f"修好的：{', '.join(c['fixed'])}")
    if c["still_failed"]:
        lines.append(f"还没修好：{', '.join(c['still_failed'])}")
    if c["regressions"]:
        lines.append(f"⚠️ 弄坏了：{', '.join(c['regressions'])}")
    if c["only_in_prev"] or c["only_in_curr"]:
        lines.append(
            f"用例集不一致：仅旧版 {c['only_in_prev'] or '无'} / 仅新版 {c['only_in_curr'] or '无'}"
        )
    return "\n".join(lines)


def main() -> None:
    if len(sys.argv) < 3:
        print("用法: python eval/compare.py 旧结果.json 新结果.json")
        raise SystemExit(1)
    prev, curr = load(sys.argv[1]), load(sys.argv[2])
    c = compare(prev, curr)
    print(to_markdown(c, os.path.basename(sys.argv[1]), os.path.basename(sys.argv[2])))


if __name__ == "__main__":
    main()
