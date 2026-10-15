"""Step 3: read the context notes, index them (TF-IDF vectors) and retrieve candidates.

Retrieval only SUGGESTS notes. Whether a note really justifies a rise is decided by
hard checks in `check_note` - similarity score alone never makes a row "justified".
"""
import re
from dataclasses import dataclass
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1)}

# Phrases meaning "this note says costs did NOT go up / nothing relevant happened"
NO_EFFECT_CUES = ["not significantly affected", "no significant", "no major", "remained normal",
                  "remained stable", "without a rate change", "not part of this dataset",
                  "returned to normal", "improved road"]
# Phrases meaning "this note describes something that pushes cost up"
RISE_CUES = ["surcharge", "higher", "rose", "pushing up", "increase", "detour", "disrupted"]
# Notes about a price level that stays changed (fuel, toll) rather than a one-off event
PERSISTENT_CUES = ["starting this week", "rose nationwide"]
PERSISTENT_WEEKS = 8   # a level change keeps showing up vs "own history" until the 8-week average absorbs it


@dataclass
class Note:
    note_id: str
    date: pd.Timestamp
    applies_to: str
    text: str
    supports_rise: bool
    win_start: pd.Timestamp
    win_end: pd.Timestamp

    @property
    def is_all_routes(self):
        return self.applies_to.strip().lower() == "all routes"


def _explicit_range(text, year):
    """'from Feb 24 to Mar 8' -> (Timestamp, Timestamp), else None."""
    m = re.search(r"from ([A-Za-z]{3})\w* (\d{1,2}) to ([A-Za-z]{3})\w* (\d{1,2})", text)
    if not m or m.group(1).lower() not in MONTHS or m.group(3).lower() not in MONTHS:
        return None
    start = pd.Timestamp(year, MONTHS[m.group(1).lower()], int(m.group(2)))
    end = pd.Timestamp(year, MONTHS[m.group(3).lower()], int(m.group(4)))
    if end < start:
        end = end.replace(year=year + 1)
    return start, end


def parse_note(row):
    text = row.note
    low = text.lower()
    date = pd.Timestamp(row.date)
    supports = any(c in low for c in RISE_CUES) and not any(c in low for c in NO_EFFECT_CUES)
    rng = _explicit_range(text, date.year)
    if rng:
        start, end = rng
    elif any(c in low for c in PERSISTENT_CUES):
        start = date
        end = date + pd.Timedelta(weeks=PERSISTENT_WEEKS) - pd.Timedelta(days=1)
    else:  # one-off event: the Monday-Sunday week the note is dated in
        start = date - pd.Timedelta(days=date.weekday())
        end = start + pd.Timedelta(days=6)
    return Note(row.note_id, date, row.applies_to.strip(), text, supports, start, end)


def load_notes(path):
    df = pd.read_csv(path)
    return [parse_note(r) for r in df.itertuples()]


class NoteIndex:
    """A tiny vector store: TF-IDF vectors for each note + cosine similarity search."""

    def __init__(self, notes):
        self.notes = notes
        self.vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
        self.matrix = self.vec.fit_transform([f"{n.applies_to}. {n.text}" for n in notes])

    def retrieve(self, route, k=4):
        """Filter by scope (this route or 'All Routes'), then rank by similarity to a query."""
        origin, dest = route.split("-", 1)
        query = f"{origin} {dest} {route} freight cost rise reason"
        sims = linear_kernel(self.vec.transform([query]), self.matrix)[0]
        scoped = [(self.notes[i], float(sims[i])) for i in range(len(self.notes))
                  if self.notes[i].is_all_routes or self.notes[i].applies_to.lower() == route.lower()]
        scoped.sort(key=lambda t: (-t[1], t[0].note_id))
        return scoped[:k]


def check_note(note, row):
    """Hard checks. Returns (ok, reason_code). All must pass for a 'justified' verdict."""
    if not note.supports_rise:
        return False, "no_effect"
    week_start = row.week_of
    week_end = week_start + pd.Timedelta(days=6)
    if note.win_end < week_start or note.win_start > week_end:
        return False, "window"
    if note.is_all_routes and row.via_peer:
        # an all-routes note hits the peers too, so it cannot explain a gap vs peers
        return False, "peer_gap"
    return True, "ok"
