"""评测运行器。

用法：
    python eval/runner.py --suite all --runs 1 --adapter mock --variant hybrid-v1
    python eval/runner.py --suite static --runs 5 --adapter live --doc <file_hash>

输出：eval/results/<suite>-<adapter>-<variant>-<时间戳>.json
每次运行都记录「原始记录」，判定在 report.py 里做 —— 这样同一份结果可以换口径重算。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from adapters import LiveAdapter, MockAdapter  # noqa: E402
from metrics import answer_pass, grounding, recall_at_k, summarize, trace_ok  # noqa: E402

SUITES = {
    "static": "datasets/qa_static.jsonl",
    "dynamic": "datasets/qa_dynamic.jsonl",
    "security": "datasets/security.jsonl",
    "fault": "datasets/fault.jsonl",
}


def load(path: str) -> List[Dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="all", choices=["all", *SUITES])
    ap.add_argument("--runs", type=int, default=1, help="同一问题重复次数（稳定性）")
    ap.add_argument("--adapter", default="mock", choices=["mock", "live"])
    ap.add_argument("--variant", default="hybrid-v1", choices=["hybrid-v1", "vector-baseline"])
    ap.add_argument("--doc", default=None, help="live 模式下绑定的文档 hash")
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    args = ap.parse_args()

    adapter = (MockAdapter(args.variant) if args.adapter == "mock"
               else LiveAdapter(document_hash=args.doc))

    cases: List[Dict] = []
    for name, path in SUITES.items():
        if args.suite in ("all", name):
            cases.extend(load(os.path.join(HERE, path)))

    records: List[Dict] = []
    t0 = time.time()
    for case in cases:
        for run in range(args.runs):
            session = f"eval-{case['id']}-{run}"
            turns = 1 + len(case.get("follow_ups", []))
            last = None
            for turn in range(turns):
                last = adapter.ask(case, turn=turn, session=session, run=run)
            rec = dict(last)
            # ---- 判定 ----
            rec["grounding"] = grounding(rec["answer"], rec["evidence"])
            rec["ok"] = answer_pass(case, rec["answer"])
            # 没有 must_contain 的「结构性用例」：改由「答案是否被证据支撑」判，
            # 否则只要答案不是空串就过，TSR 会虚高 —— 这是最容易被自己骗的地方
            if not case.get("must_contain") and not case.get("should_refuse"):
                rec["ok"] = rec["ok"] and rec["grounding"] >= 0.5
            # 故障用例：判「有没有降级得体面」，而不是判内容对不对
            if case.get("suite") == "fault":
                graceful = any(m in rec["answer"] for m in ("不可用", "无响应", "没有找到", "请稍后"))
                rec["ok"] = bool(rec["answer"].strip()) and graceful
            rec["recall"] = recall_at_k(rec["retrieved_ids"], rec["expected_evidence"], k=3)
            rec["process_ok"] = trace_ok(rec)
            rec["turn"] = turns
            rec["run"] = run
            records.append(rec)
        done = len(records)
        print(f"\r  {case['id']:<5} {done}/{len(cases) * args.runs}", end="", flush=True)
    print()

    stamp = time.strftime("%Y%m%d-%H%M%S")
    os.makedirs(args.out, exist_ok=True)
    out = os.path.join(args.out, f"{args.suite}-{args.adapter}-{args.variant}-{stamp}.json")
    payload = {
        "suite": args.suite,
        "adapter": args.adapter,
        "variant": args.variant,
        "runs_per_case": args.runs,
        "cases": len(cases),
        "elapsed_s": round(time.time() - t0, 2),
        "summary": summarize(records),
        "records": records,
    }
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    s = payload["summary"]
    print(f"\n结果 -> {out}")
    print(f"  TSR {s['tsr']:.1%} | 过程成功率 {s['process_ok']:.1%} | "
          f"证据支撑度 {s['grounding']:.2f} | recall@3 {s['recall_at_k']:.2f} | "
          f"p50 {s['cost']['p50_ms']:.0f}ms / p95 {s['cost']['p95_ms']:.0f}ms")
    if args.adapter == "mock":
        print("  ⚠️ 这是 mock 数据，只验证评测链路，不能当作真实效果引用")


if __name__ == "__main__":
    main()
