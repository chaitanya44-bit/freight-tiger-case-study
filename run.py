"""Cost-watch assistant: python run.py   (add --llm to polish reasons with a local Ollama model)"""
import argparse
import csv
import json
import os
import pandas as pd
from costwatch.weekly import load_shipments, weekly_table, add_baselines
from costwatch.flagging import detect_rises
from costwatch.notes import load_notes, NoteIndex
from costwatch.explain import decide, build_reason, make_row, llm_polish, ollama_call, LLMStats

COLUMNS = ["route", "week_of", "cost_per_tonne_km", "vs_own_history", "vs_similar_routes",
           "flagged", "matched_note_id", "reason"]


def run(shipments_path, notes_path, out_dir, use_llm=False):
    os.makedirs(out_dir, exist_ok=True)
    weekly = detect_rises(add_baselines(weekly_table(load_shipments(shipments_path))))
    index = NoteIndex(load_notes(notes_path))
    stats = LLMStats()

    rows = []
    for r in weekly[weekly["is_rise"]].itertuples():
        note, rejected = decide(r, index)
        reason = build_reason(r, note, rejected)
        if use_llm and note:
            reason = llm_polish(reason, note, lambda p: ollama_call(p, stats))
        rows.append(make_row(r, note, reason))
    out = pd.DataFrame(rows, columns=COLUMNS).sort_values(["week_of", "route"])

    out.to_csv(os.path.join(out_dir, "flagged_output.csv"), index=False, quoting=csv.QUOTE_MINIMAL)
    # full weekly table for anyone who wants to check the numbers
    cols = ["route", "route_type", "week_of", "shipments", "total_cost", "tonne_km", "cost_per_tonne_km",
            "own_avg", "n_prior_weeks", "peer_avg", "own_pct", "peer_pct", "via_own", "via_peer", "is_rise"]
    weekly[cols].round({"cost_per_tonne_km": 4, "own_avg": 4, "peer_avg": 4, "own_pct": 4, "peer_pct": 4, "tonne_km": 1}).to_csv(os.path.join(out_dir, "weekly_metrics.csv"), index=False)

    log = {"mode": "template + local Ollama rewrite (temperature 0)" if use_llm else "template only (no LLM used)",
           "llm_calls": stats.calls, "input_tokens": stats.tokens_in, "output_tokens": stats.tokens_out,
           "estimated_cost_usd": 0.0, "note": "open-source local model / no paid API"}
    with open(os.path.join(out_dir, "token_cost_log.json"), "w") as f:
        json.dump(log, f, indent=2)
    return out, weekly, log


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--shipments", default="data/shipment_records.csv")
    ap.add_argument("--notes", default="data/context_notes.csv")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--llm", action="store_true")
    a = ap.parse_args()
    out, weekly, log = run(a.shipments, a.notes, a.out, a.llm)
    print(f"{len(weekly)} route-weeks checked, {len(out)} cost rises found "
          f"({(out.flagged == 'Yes').sum()} unexplained, {(out.flagged != 'Yes').sum()} justified)")
    print("LLM log:", log)
