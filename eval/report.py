"""把评测结果渲染成 markdown 报告（七维）。

用法：
    python eval/report.py eval/results/a.json [eval/results/b.json ...] > eval/results/report.md

设计：
  · 一次可以传多个结果文件 —— 多版本对比表是报告的主体（「和谁比」比「多少分」重要）；
  · 每个维度只给一行结论 + 一个数字，长解释放到文末；
  · 失败用例逐条列原因，因为报告是给自己看的，不是给评委看的。
"""
from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def load(path: str) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def label(d: Dict[str, Any]) -> str:
    return f"{d['variant']}（{d['adapter']}）"


def fmt_pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def main() -> None:
    paths = sys.argv[1:]
    if not paths:
        print("用法: python eval/report.py <结果.json> [更多结果.json]")
        raise SystemExit(1)
    runs = [load(p) for p in paths]
    base, *others = runs

    out: List[str] = []
    out.append("# 知渊 · Agent 评测报告\n")
    # 数据来源声明：没有这句，报告就会被当成真实效果引用
    if all(r["adapter"] == "mock" for r in runs):
        out.append("> ⚠️ **本报告由 mock 适配器生成**：数字是模拟出来的，只用于验证「评测链路能不能分辨好坏」，"
                   "**不可用于简历、答辩或对外宣传**。真实数字请用 `--adapter live` 重跑。\n")
    out.append(f"\n- 用例：{base['cases']} 条 × {base['runs_per_case']} 次"
               f"（suite={base['suite']}）\n- 生成时间：{os.path.basename(paths[0])}\n")

    # ---------- 一、结论 ----------
    out.append("\n## 一、结论：和谁比、差多少\n")
    head = ["指标"] + [label(r) for r in runs]
    out.append("| " + " | ".join(head) + " |")
    out.append("|" + "---|" * len(head))
    rows = [
        ("任务成功率 TSR", lambda r: fmt_pct(r["summary"]["tsr"])),
        ("过程成功率（无异常/超时/5xx）", lambda r: fmt_pct(r["summary"]["process_ok"])),
        ("证据支撑度（答案被原文支撑）", lambda r: f"{r['summary']['grounding']:.2f}"),
        ("检索 recall@3", lambda r: f"{r['summary']['recall_at_k']:.2f}"),
        ("稳定性 pass@n / pass^n", lambda r: f"{r['summary']['stability']['pass_at_n']:.0%} / "
                                            f"{r['summary']['stability']['pass_all_n']:.0%}"),
        ("答案一致性", lambda r: fmt_pct(r["summary"]["stability"]["consistency"])),
        ("安全拦截率", lambda r: fmt_pct(r["summary"]["security"]["blocked_rate"])),
        ("故障降级正确率", lambda r: fmt_pct(r["summary"].get("degrade", 0.0))),
        ("p50 / p95 耗时", lambda r: f"{r['summary']['cost']['p50_ms']:.0f} / "
                                    f"{r['summary']['cost']['p95_ms']:.0f} ms"),
        ("平均 token", lambda r: f"{r['summary']['cost']['avg_tokens']:.0f}"),
    ]
    for name, fn in rows:
        out.append("| " + name + " | " + " | ".join(fn(r) for r in runs) + " |")

    if len(runs) >= 2:
        d_tsr = others[0]["summary"]["tsr"] - base["summary"]["tsr"]
        d_lat = others[0]["summary"]["cost"]["p50_ms"] - base["summary"]["cost"]["p50_ms"]
        out.append(f"\n> 对比口径：以 {label(base)} 为基线，{label(others[0])} "
                   f"TSR {d_tsr:+.1%}、p50 {d_lat:+.0f} ms（负值表示更快）。\n")

    # ---------- 二、七维明细 ----------
    out.append("\n## 二、七维明细\n")
    out.append("| 维度 | 这个数字从哪来 | 本次结果 |")
    out.append("|---|---|---|")
    s = base["summary"]
    out.append(f"| 1 结果 | 用例的 `must_contain` 全部命中、`forbidden` 一个不出现、拒答类命中拒答话术 | "
               f"TSR {fmt_pct(s['tsr'])} |")
    out.append(f"| 2 检索 | 返回条数是否落在 `k_range`、期望片段是否进 recall@3 | "
               f"recall@3 {s['recall_at_k']:.2f} |")
    out.append(f"| 3 证据 | 答案按句切，逐句与证据比 token 重叠，≥0.5 才算被支撑 | "
               f"{s['grounding']:.2f} |")
    out.append(f"| 4 过程 | trace 里出现 error / timeout / 5xx 即判失败 | "
               f"{fmt_pct(s['process_ok'])} |")
    out.append(f"| 5 稳定性 | 同题跑 {base['runs_per_case']} 次：pass@n 与 pass^n 分开看 | "
               f"{s['stability']['pass_at_n']:.0%} / {s['stability']['pass_all_n']:.0%} |")
    out.append(f"| 6 成本 | 每次调用的耗时与 token 粗估（中文 1.6 字/token） | "
               f"p50 {s['cost']['p50_ms']:.0f} ms，均 {s['cost']['avg_tokens']:.0f} tok |")
    out.append(f"| 7 安全 | 越权 / 注入 / 超范围用例的拒答率 | "
               f"{fmt_pct(s['security']['blocked_rate'])} |")

    # ---------- 三、失败清单 ----------
    fails = [r for r in base["records"] if not r["ok"] or not r["process_ok"]]
    out.append(f"\n## 三、失败清单（{len(fails)} 条）\n")
    if not fails:
        out.append("无。\n")
    else:
        out.append("| 用例 | 问题 | 判定 | 答案开头 |")
        out.append("|---|---|---|---|")
        for r in fails[:30]:
            reasons = []
            if not r["ok"]:
                reasons.append("答案不达标")
            if not r["process_ok"]:
                reasons.append("过程异常（" + (r.get("error") or "trace 异常") + "）")
            if r["expected_evidence"] and r["grounding"] < 0.5:
                reasons.append("证据支撑不足")
            out.append(f"| {r['case_id']} | {r['question'][:22]} | {'；'.join(reasons)} | "
                       f"{(r['answer'] or '（空）')[:26]} |")

    # ---------- 四、局限 ----------
    review = len([r for r in base["records"] if r.get("needs_review")])
    out.append("\n## 四、局限与下一步\n")
    out.append(f"- 有 **{review} 条**用例的事实断言标了 `needs_review`：仓库不含知识库 PDF，"
               "这些 `must_contain` 需要按真实语料核校后才能当准数。")
    out.append("- 证据支撑度是 token 重叠的**下界**，不理解同义改写，会低估真实忠实度；"
               "线上应叠加人工抽检。")
    out.append("- token 是粗估（按字符折算），只能比较版本之间的相对变化，不能当账单。")
    out.append("- mock 数据只验证链路；真实结论必须 `--adapter live` 跑，并固定同一份语料与同一批用例。")

    print("\n".join(out))


if __name__ == "__main__":
    main()
