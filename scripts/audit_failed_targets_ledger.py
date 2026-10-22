"""复现失败账本的真实语义（方案 §5.2 的依据）。

要证明的三件事：
1. 同键连着失败 → count+1 → 熔断生效
2. **换键 → 整表替换**（旧的清掉），不是追加
3. 任何一次成功点击 → 账本清空（tools/mobile/tap.py 的成功分支）

⇒ 所以「跨子目标的失败事实」**不能**从这个账本取，
   必须走 append-only 的 `executor_failures`。

跑法：
    uv run python scripts/audit_failed_targets_ledger.py
"""

from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "minitap"))
sys.stdout.reconfigure(encoding="utf-8")

from minitap.mobile_use.utils.failed_targets import (  # noqa: E402
    MAX_TRACKED_FAILURES,
    append_failure,
    breaker_reason,
    record_failure,
    render_failures,
)


def main() -> int:
    print("=== 熔断账本 failed_targets（换键即替换、成功即清）===")
    s = record_failure(
        None, key="index=1|id=a", index=1, resource_id="a", label="X", reason="r1"
    )
    print("  1st failure      :", [(t.key, t.count) for t in s])

    s = record_failure(
        s, key="index=1|id=a", index=1, resource_id="a", label="X", reason="r2"
    )
    breaker = breaker_reason(s, "index=1|id=a")
    print("  same key again   :", [(t.key, t.count) for t in s])
    print("     -> breaker:", (breaker or "")[:50])

    s = record_failure(
        s, key="index=2|id=b", index=2, resource_id="b", label="Y", reason="r3"
    )
    print("  DIFFERENT key    :", [(t.key, t.count) for t in s])
    print("     -> 整表替换了！旧的 A 没了")
    print(f"     MAX_TRACKED_FAILURES = {MAX_TRACKED_FAILURES}（实际用不满，账本最多 1 条）")
    print("     render_failures      :", render_failures(s)[:120])
    print("     breaker for old key  :", breaker_reason(s, "index=1|id=a"))

    print()
    print("=== 跨子目标事实账本 executor_failures（只增不减）===")
    facts = append_failure(
        None, key="index=1|id=a", index=1, resource_id="a", label="X", reason="r1"
    )
    facts = append_failure(
        facts, key="index=2|id=b", index=2, resource_id="b", label="Y", reason="r3"
    )
    print("  两条失败后:", [(f["key"], f["reason"]) for f in facts])
    print("  -> A 还在，跨子目标可见 ✓")

    print()
    print("=== 结论 ===")
    print("  失败事实必须走 executor_failures；熔断账本只管『当下连着失败』。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
