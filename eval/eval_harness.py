"""Run with:  python eval/eval_harness.py   (from the project root)

Five independent checks - each one exists because it can catch a different kind of mistake.
  1. Maths    : re-compute every weekly number with plain csv/datetime code (no pandas) and compare.
  2. Sample   : the 3 example rows in sample_output_format.csv must come out with identical numbers.
  3. Labelled : 19 route-weeks I labelled by hand (reading the notes + the weekly numbers).
  4. Oracle   : every 'justified' row is checked against a hand-typed table of what each note can cover.
  5. Stress   : inject fake cost spikes into a copy of the data and check the verdict is right
                (including wrong-route, wrong-week and 'no effect' notes), plus LLM-guard tests.
"""
import csv
import os
import sys
import tempfile
from collections import defaultdict
from datetime import date, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
from run import run
from costwatch.notes import load_notes
from costwatch.explain import llm_polish

SHIP, NOTES = "data/shipment_records.csv", "data/context_notes.csv"
passed = failed = 0


def check(name, ok, detail=""):
    global passed, failed
    passed += ok
    failed += (not ok)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail}")


def monday(d):
    return d - timedelta(days=d.weekday())


# ---------- 1. independent re-computation of the weekly maths ----------
def independent_weekly():
    cost, tk, rtype = defaultdict(float), defaultdict(float), {}
    with open(SHIP) as f:
        for r in csv.DictReader(f):
            wk = monday(date.fromisoformat(r["shipment_date"]))
            key = (r["origin"] + "-" + r["destination"], wk)
            cost[key] += float(r["freight_cost_inr"])
            tk[key] += float(r["quantity_tonnes"]) * float(r["distance_km"])
            rtype[key[0]] = r["route_type"]
    cpt = {k: cost[k] / tk[k] for k in cost}
    res = {}
    for (route, wk), v in cpt.items():
        prior = [cpt[(route, wk - timedelta(weeks=i))] for i in range(1, 9) if (route, wk - timedelta(weeks=i)) in cpt]
        peers = [cpt[(r, wk)] for r in rtype if r != route and rtype[r] == rtype[route] and (r, wk) in cpt]
        res[(route, wk)] = (v, sum(prior) / len(prior) if prior else None, sum(peers) / len(peers) if peers else None)
    return res


def test_maths(weekly):
    print("1. Independent maths re-computation")
    ref = independent_weekly()
    worst, n = 0.0, 0
    for r in weekly.itertuples():
        v, own, peer = ref[(r.route, r.week_of.date())]
        diffs = [abs(v - r.cost_per_tonne_km)]
        if own is not None:
            diffs.append(abs(own - r.own_avg))
        if peer is not None:
            diffs.append(abs(peer - r.peer_avg))
        worst, n = max(worst, *diffs), n + 1
    check(f"{n} route-weeks match", n == len(ref) and worst < 1e-9, f"(max abs diff {worst:.2e})")


# ---------- 2. the example rows from the brief ----------
def test_sample(out):
    print("2. Sample-format rows")
    # the 3rd sample row has an unquoted comma in `reason`, so read the first 6 fields by hand
    with open("data/sample_output_format.csv") as f:
        rows = [line.rstrip("\n").split(",")[:6] for line in f][1:]
    for route, wk, cpt, own, peer, flag in rows:
        mine = out[(out.route == route) & (out.week_of == wk)]
        ok = (len(mine) == 1 and str(mine.iloc[0].cost_per_tonne_km) == str(float(cpt))
              and mine.iloc[0].vs_own_history == own and mine.iloc[0].vs_similar_routes == peer
              and mine.iloc[0].flagged == flag)
        check(f"{route} {wk}", ok)


# ---------- 3. hand-labelled set ----------
def test_labelled(out):
    print("3. Hand-labelled set")
    lab = pd.read_csv("eval/labeled_set.csv", keep_default_na=False)
    right = 0
    tp = fp = fn = 0
    for r in lab.itertuples():
        row = out[(out.route == r.route) & (out.week_of == r.week_of)]
        got = row.iloc[0].flagged if len(row) else "not flagged"
        note = row.iloc[0].matched_note_id if len(row) else ""
        ok = (got == r.expected) and (note == r.expected_note)
        right += ok
        if not ok:
            print(f"      MISS {r.route} {r.week_of}: expected {r.expected}/{r.expected_note!r}, got {got}/{note!r}")
        rise_expected, rise_got = r.expected != "not flagged", got != "not flagged"
        tp += rise_expected and rise_got
        fp += (not rise_expected) and rise_got
        fn += rise_expected and (not rise_got)
    check(f"verdict + note correct on {right}/{len(lab)}", right == len(lab))
    print(f"      rise-detection on labelled set: precision {tp/(tp+fp):.2f}, recall {tp/(tp+fn):.2f}")


# ---------- 4. oracle: what each note is allowed to justify (typed by hand from reading the notes) ----------
ORACLE = {  # note_id: (routes or None for all, first day, last day)
    "N001": ({"Chennai-Bangalore"}, date(2025, 2, 24), date(2025, 3, 8)),
    "N002": ({"Ahmedabad-Mumbai"}, date(2025, 1, 20), date(2025, 1, 26)),
    "N003": (None, date(2025, 5, 5), date(2025, 6, 29)),
}


def audit_justified(out):
    bad = []
    for r in out[out.flagged == "No (justified)"].itertuples():
        rule = ORACLE.get(r.matched_note_id)
        wk = date.fromisoformat(r.week_of)
        if (rule is None or (rule[0] and r.route not in rule[0]) or rule[2] < wk or rule[1] > wk + timedelta(days=6)):
            bad.append((r.route, r.week_of, r.matched_note_id))
        if r.matched_note_id not in r.reason:
            bad.append((r.route, r.week_of, "id missing in reason"))
    return bad


def test_oracle(out):
    print("4. Hallucination audit (oracle)")
    n = int((out.flagged == "No (justified)").sum())
    bad = audit_justified(out)
    check(f"all {n} justified rows cite a note valid for that route AND week", not bad, str(bad) if bad else "")
    check("rows with no match have blank matched_note_id", (out[out.flagged == "Yes"].matched_note_id == "").all())


# ---------- 5. stress tests ----------
def inject(route, week, factor):
    df = pd.read_csv(SHIP, parse_dates=["shipment_date"])
    m = ((df.origin + "-" + df.destination) == route) & \
        ((df.shipment_date - pd.to_timedelta(df.shipment_date.dt.weekday, unit="D")) == pd.Timestamp(week))
    assert m.any(), f"no shipments for {route} {week}"
    df.loc[m, "freight_cost_inr"] = (df.loc[m, "freight_cost_inr"] * factor).round().astype(int)
    path = tempfile.mktemp(suffix=".csv")
    df.to_csv(path, index=False)
    out, _, _ = run(path, NOTES, tempfile.mkdtemp())
    os.remove(path)
    return out[(out.route == route) & (out.week_of == week)]


def test_stress():
    print("5a. Injected cost spikes")
    cases = [  # route, week, factor, expected flag, expected note, what this proves
        ("Kolkata-Bhubaneswar", "2024-06-10", 1.30, "Yes", "", "no note exists -> must stay unexplained"),
        ("Kolkata-Bhubaneswar", "2025-06-09", 1.30, "Yes", "", "N008 is dated that week but says 'no disruptions'"),
        ("Mumbai-Pune", "2025-01-20", 1.30, "Yes", "", "N002 is the same week but for ANOTHER route"),
        ("Chennai-Bangalore", "2025-04-14", 1.30, "Yes", "", "flood note exists but the window is long over"),
        ("Ahmedabad-Mumbai", "2025-01-27", 1.30, "Yes", "", "festival note covers only the week before"),
        ("Delhi-Chennai", "2025-05-19", 1.13, "No (justified)", "N003", "moderate rise inside the diesel window -> justified"),
        ("Delhi-Chennai", "2025-05-19", 1.30, "Yes", "", "huge gap vs peers - diesel hits peers too, so no"),
    ]
    for route, week, f, flag, note, why in cases:
        row = inject(route, week, f)
        got = (row.iloc[0].flagged, row.iloc[0].matched_note_id) if len(row) else ("not flagged", "")
        check(f"{route} {week} x{f}: {why}", got == (flag, note), f"-> got {got}")


def test_llm_guard():
    print("5b. LLM guard (fake model, no network)")
    note = next(n for n in load_notes(NOTES) if n.note_id == "N002")
    base = "Matches note N002 dated 2025-01-20: festival surcharge. The cost rise has a clear explanation."
    good = "Note N002 (2025-01-20) explains it: a festival surcharge."
    check("faithful rewrite is accepted", llm_polish(base, note, lambda p: good) == good)
    check("invented note id is rejected", llm_polish(base, note, lambda p: "Because of N005 (2025-01-20).") == base)
    check("invented number is rejected", llm_polish(base, note, lambda p: "N002 added a 12% surcharge.") == base)
    check("missing note id is rejected", llm_polish(base, note, lambda p: "A festival caused it.") == base)

    def broken(p):
        raise ConnectionError("no server")
    check("LLM outage falls back to template", llm_polish(base, note, broken) == base)


if __name__ == "__main__":
    out, weekly, _ = run(SHIP, NOTES, tempfile.mkdtemp())
    test_maths(weekly)
    test_sample(out)
    test_labelled(out)
    test_oracle(out)
    test_stress()
    test_llm_guard()
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
