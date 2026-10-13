"""Audita gerações salvas sem alterar o cache ou os rankings originais."""
import argparse
import json
from pathlib import Path

import pandas as pd

from .generation import qexp_parse_generation


def audit_generations(records, count=4):
    rows = []
    for record in records:
        kind = record["specification"]["kind"]
        if kind != "multi":
            continue
        outcome = {"query": record["query"], "cache_key": record["cache_key"],
                   "original_status": record["status"], "raw": record["raw"]}
        for policy in ("strict_json", "numbered"):
            try:
                qexp_parse_generation(record["raw"], record["query"], kind, count, policy)
                outcome[policy + "_valid"] = True
                outcome[policy + "_error"] = ""
            except ValueError as error:
                outcome[policy + "_valid"] = False
                outcome[policy + "_error"] = str(error)
        rows.append(outcome)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=4)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Use um arquivo de saída novo.")
    frame = audit_generations(json.loads(args.generations.read_text()), args.count)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    if not frame.empty:
        print(frame.groupby(["original_status", "strict_json_valid", "numbered_valid"]).size())


if __name__ == "__main__":
    main()
