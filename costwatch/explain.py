"""Step 4: turn a verdict into the output row + plain-English reason.

The reason text is built from the note's own words (template), so it can't invent a cause.
An optional local-LLM rewrite exists (--llm) but it can never change the verdict or the
cited note, and it is thrown away if it adds anything not in the original text.
"""
import json
import re
import urllib.request
import pandas as pd
from .notes import check_note

NEARBY_DAYS = 21      # only mention a rejected note if it is dated this close to the week
UNEXPLAINED = "No matching note found for this route or date range. Cost rise looks unexplained and worth a human review."
WHY = {
    "no_effect": "does not describe a reason for a cost rise on this route",
    "window": "has an effect window ({s} to {e}) that does not cover this week",
    "peer_gap": "applies to all routes equally, so it cannot explain a gap against similar routes",
}


def first_sentence(text):
    return re.split(r"(?<=[.;])\s", text.strip())[0].rstrip(".")


def decide(row, index):
    """Returns (matched_note or None, [(rejected_note, why_code), ...])."""
    passed, rejected = [], []
    for note, score in index.retrieve(row.route):
        ok, code = check_note(note, row)
        if ok:
            passed.append((note, score))
        else:
            rejected.append((note, code))
    # prefer a route-specific note over an all-routes one, then higher similarity
    passed.sort(key=lambda t: (t[0].is_all_routes, -t[1], t[0].note_id))
    return (passed[0][0] if passed else None), rejected


def build_reason(row, note, rejected):
    if note:
        return (f"Matches note {note.note_id} dated {note.date.date()}: {first_sentence(note.text)}. "
                f"Effect window {note.win_start.date()} to {note.win_end.date()} covers this week. "
                f"The cost rise has a clear explanation.")
    week_start = row.week_of
    week_end = week_start + pd.Timedelta(days=6)
    gap = pd.Timedelta(days=NEARBY_DAYS)
    # mention a rejected note if it is dated close to the week OR its effect window touches the week
    near = [(n, c) for n, c in rejected
            if (week_start - gap <= n.date <= week_end + gap) or (n.win_start <= week_end and n.win_end >= week_start)]
    if not near:
        return UNEXPLAINED
    parts = [f"{n.note_id} ({n.date.date()}) " + WHY[c].format(s=n.win_start.date(), e=n.win_end.date())
             for n, c in near[:2]]
    label = "The closest note" if len(parts) == 1 else "The closest notes"
    return f"{label}: " + "; ".join(parts) + ". No genuine justification found; flagged for review."


def pct(x, what):
    return "n/a" if pd.isna(x) else f"{x * 100:+.1f}% vs {what}"


def make_row(row, note, reason):
    return {
        "route": row.route,
        "week_of": row.week_of.strftime("%Y-%m-%d"),
        "cost_per_tonne_km": round(row.cost_per_tonne_km, 2),
        "vs_own_history": pct(row.own_pct, "this route's past average"),
        "vs_similar_routes": pct(row.peer_pct, "similar-length routes this week"),
        "flagged": "No (justified)" if note else "Yes",
        "matched_note_id": note.note_id if note else "",
        "reason": reason,
    }


# ---------------- optional LLM polish (off by default) ----------------
class LLMStats:
    def __init__(self):
        self.calls = self.tokens_in = self.tokens_out = 0


def ollama_call(prompt, stats, model="llama3.2:3b", url="http://localhost:11434/api/generate"):
    """Local Ollama model, temperature 0 + fixed seed so repeated runs give the same text."""
    body = json.dumps({"model": model, "prompt": prompt, "stream": False,
                       "options": {"temperature": 0, "seed": 42}}).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read())
    stats.calls += 1
    stats.tokens_in += data.get("prompt_eval_count", 0)
    stats.tokens_out += data.get("eval_count", 0)
    return data["response"].strip()


def llm_polish(reason, note, call):
    """Ask the model to rephrase `reason` in 1-2 plain sentences. Fall back to the template
    if the rewrite drops the note id or introduces any other note id or number."""
    prompt = ("Rewrite this freight-cost explanation in one or two plain sentences. "
              "Use ONLY the facts given. Keep the note id and dates exactly. Add nothing new.\n\n"
              f"Note {note.note_id}: {note.text}\nExplanation: {reason}")
    try:
        new = call(prompt)
    except Exception as exc:  # no Ollama running etc.
        print(f"[warn] LLM unavailable ({exc}); using template text")
        return reason
    allowed = set(re.findall(r"\d[\d.\-%]*", reason + " " + note.text))
    if note.note_id not in new or set(re.findall(r"N\d{3}", new)) - {note.note_id}:
        return reason
    if set(re.findall(r"\d[\d.\-%]*", new)) - allowed:
        return reason
    return new
