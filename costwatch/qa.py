"""Stretch goal: ask questions about the findings in plain English.

Answers are built only from the output CSV rows - nothing is generated from thin air.
Example:  python -m costwatch.qa "why did Chennai-Bangalore get pricier in March 2025?"
"""
import re
import sys
import pandas as pd

MONTHS = ["january", "february", "march", "april", "may", "june", "july",
          "august", "september", "october", "november", "december"]


def answer(question, results):
    q = question.lower()
    df = results.copy()
    df["week"] = pd.to_datetime(df["week_of"])

    routes = [r for r in df["route"].unique() if r.lower() in q]
    if not routes:  # allow just a city name, e.g. "Pune" or "Jaipur"
        routes = [r for r in df["route"].unique() if any(c.lower() in q for c in r.split("-"))]
    month = next((i + 1 for i, m in enumerate(MONTHS) if m in q or re.search(rf"\b{m[:3]}\b", q)), None)
    year = re.search(r"\b(20\d\d)\b", q)

    if routes:
        df = df[df["route"].isin(routes)]
    if month:
        df = df[df["week"].dt.month == month]
    if year:
        df = df[df["week"].dt.year == int(year.group(1))]
    if "unexplained" in q or "no reason" in q or "review" in q:
        df = df[df["flagged"] == "Yes"]

    if df.empty:
        return "I found no flagged cost rise matching that question in my results. " \
               "That means no week crossed my flag thresholds - it does not prove costs were flat."
    lines = [f"Found {len(df)} week(s) with a flagged or justified cost rise:"]
    for r in df.sort_values(["route", "week"]).itertuples():
        lines.append(f"- {r.route}, week of {r.week_of}: {r.cost_per_tonne_km} INR/tonne-km "
                     f"({r.vs_own_history}; {r.vs_similar_routes}). [{r.flagged}"
                     f"{', note ' + r.matched_note_id if isinstance(r.matched_note_id, str) and r.matched_note_id else ''}] {r.reason}")
    return "\n".join(lines)


if __name__ == "__main__":
    path = "outputs/flagged_output.csv"
    res = pd.read_csv(path, keep_default_na=False)
    question = " ".join(sys.argv[1:]) or input("Ask a question: ")
    print(answer(question, res))
