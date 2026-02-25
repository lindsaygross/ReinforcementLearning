import json
from statistics import mean
from typing import Any, Dict, List


def build_training_rows(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for record in records:
        if record.get("preference") == "tie":
            continue
        chosen = record.get("chosen")
        rejected = record.get("rejected")
        prompt = record.get("prompt")
        if prompt is None or chosen is None or rejected is None:
            continue
        rows.append(
            {
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected,
            }
        )
    return rows


def to_jsonl(rows: List[Dict[str, Any]]) -> str:
    return "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)


def compute_stats(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(records)
    pref_a = sum(1 for r in records if r.get("preference") == "a")
    pref_b = sum(1 for r in records if r.get("preference") == "b")
    ties = sum(1 for r in records if r.get("preference") == "tie")

    a_lat = [r.get("response_a_latency_ms") for r in records if isinstance(r.get("response_a_latency_ms"), (int, float))]
    b_lat = [r.get("response_b_latency_ms") for r in records if isinstance(r.get("response_b_latency_ms"), (int, float))]

    return {
        "total": total,
        "pref_a": pref_a,
        "pref_b": pref_b,
        "ties": ties,
        "avg_a_latency_ms": round(mean(a_lat), 2) if a_lat else 0,
        "avg_b_latency_ms": round(mean(b_lat), 2) if b_lat else 0,
    }
