# Shipping Cost Watch - FreightTiger case study

A small assistant that looks at weekly freight cost per tonne-km for each route, spots rises
that look out of the ordinary, and then checks the context notes to see if there is a *real*
reason. If it can't find one, it says so instead of guessing.

## How to run

```bash
pip install -r requirements.txt          # pandas, scikit-learn
python run.py                            # writes outputs/flagged_output.csv (+ weekly_metrics.csv, token log)
python eval/eval_harness.py              # all checks (19 pass)
python repro_check.py                    # runs 3 times, compares file hashes
python -m costwatch.qa "why did Chennai-Bangalore get pricier in March?"   # stretch goal
```

## What it does (in order)

| Step | File | What happens |
|---|---|---|
| 1 | `costwatch/weekly.py` | Route = origin-destination. `week_of` = Monday of the shipment's week. Weekly cost per tonne-km = **sum(cost) / sum(tonnes x km)** for that route-week. Then the two baselines, exactly as the brief defines them. |
| 2 | `costwatch/flagging.py` | Marks a route-week as a *cost rise* if it is above its own history **and** (≥ +10% vs own history **or** ≥ +20% vs similar routes). |
| 3 | `costwatch/notes.py` | Notes are turned into vectors (TF-IDF). For a flagged row I filter notes to that route + "All Routes", rank them by similarity, then run hard checks on the top candidates. |
| 4 | `costwatch/explain.py` | Builds the verdict and the reason from the matched note's own words. |
| 5 | `costwatch/qa.py` | Answers plain-English questions using only the rows in the output file. |

### Baselines (as the brief asks)
- **vs own history**: mean of this route's weekly values over the 8 calendar weeks *strictly before* the week. If fewer than 8 prior weeks exist I use what exists (no padding). That only happens in the first 8 weeks of 2024.
- **vs similar routes**: mean of the *other* routes' weekly value, same week, same `route_type`. The route itself is excluded.
- Percentages are worked out from unrounded values; `cost_per_tonne_km` in the output is rounded to 2 decimals.

### Why two baselines are needed
Mumbai-Pune creeps up slowly (about 3.4 in early 2025 to 4.5 by December). An 8-week average chases a slow creep, so
"vs own history" only shows +4% to +9%. Against similar routes it is +20% to +39%. Using only one baseline would miss this.

### When is a note allowed to say "justified"?
Retrieval only *suggests* notes. A note must pass all of these checks, otherwise the row stays **Yes** (unexplained):

1. **Route**: note is for this route, or says "All Routes".
2. **Effect**: note actually says costs went up. Notes like "costs were not significantly affected", "no major disruptions", "without a rate change", "returned to normal" never count.
3. **Window**: the note's effect window overlaps the Monday-Sunday week. Explicit dates in the text are used ("from Feb 24 to Mar 8"); a one-off event covers its own week; a price-level change ("diesel rose starting this week") covers 8 weeks, because after 8 weeks the own-history average has absorbed the new level.
4. **All-routes notes can't explain a peer gap**: a diesel rise hits the similar routes too, so it cannot be the reason a route is +30% *above* them.

`matched_note_id` is filled only if all four pass, otherwise it is blank. If a nearby note was rejected, the reason says which one and why (like the sample file does for N006).

## Results (full 2024-2025 file: 728 route-weeks, 20 cost rises)

- **4 justified**: Ahmedabad-Mumbai 2025-01-20 (N002), Chennai-Bangalore 2025-02-24 and 2025-03-03 (N001), Mumbai-Delhi 2025-05-12 (N003).
- **16 unexplained**:
  - Delhi-Jaipur 2024-11-11 and 2024-11-18 (+35% / +28%, no note).
  - Chennai-Bangalore 2025-03-10 and 03-17: costs stayed ~+22% / +14% *after* the flood window ended on Mar 8 and N009 says things were normal. Worth a human look.
  - Mumbai-Pune, 12 weeks between June and December 2025: slow creep with no matching note. This is the biggest finding.
- First 4 weeks of 2024 are never flagged (less than 4 weeks of history). Numbers are still in `weekly_metrics.csv`.

The three example rows in `sample_output_format.csv` come out with identical numbers and verdicts (checked automatically).

## How I convinced myself it is right

`python eval/eval_harness.py` runs five checks:

1. **Maths re-done separately.** I re-wrote the weekly maths with plain `csv`/`datetime` (no pandas) and compared all 728 route-weeks (cost, own average, peer average). Biggest difference: 9e-16.
2. **Sample rows** from the brief match.
3. **Hand-labelled set** (`eval/labeled_set.csv`, 19 route-weeks, 100% match). Honest caveat: I wrote these labels from reading the notes and the weekly numbers, and I had already seen the pipeline output, so this is more of a regression test than an independent test.
4. **Oracle audit.** A hand-typed table says what each note may cover (route + dates). Every justified row is checked against it, and notes N004-N010 may never be cited.
5. **Stress tests.** I inject fake +30% spikes into a copy of the data and check the verdict: no note exists, note for another route in the same week, note that says "no disruptions", flood note after its window, festival note one week late. All must stay unexplained. A moderate rise inside the diesel window must be justified (N003), a huge one must not.

Threshold sensitivity (how many rows get flagged):

| own / peer threshold | 15% peer | 20% peer | 25% peer |
|---|---|---|---|
| 8% own | 28 | 21 | 16 |
| **10% own** | 27 | **20** | 14 |
| 12% own | 26 | 19 | 13 |

The justified count stays at 4 (3 at 12%), so the verdicts don't depend much on where I put the line. The chosen values
sit just above normal noise in Jan-Oct 2024 (max +8.5% vs own history; usual peer gap under 17%). Thresholds are my judgement, not learned.

## Reproducibility

`python repro_check.py` runs the pipeline 3 times with different `PYTHONHASHSEED` values and compares SHA-256 hashes of the output files.
Result (see `outputs/repro_log.txt`): **byte-identical**. The default run has nothing random: no LLM, sorted data, fixed tie-breaks (similarity, then note id).

## Cost of running it (one full run)

| | |
|---|---|
| LLM calls | **0** |
| Input tokens | 0 |
| Output tokens | 0 |
| Cost | $0 (no paid API; TF-IDF runs locally) |

(`outputs/token_cost_log.json`). The reasons are built from the note text, so they cannot invent a cause.
There is an optional `python run.py --llm` that asks a local Ollama model (`llama3.2:3b`, temperature 0, seed 42) to rephrase the reason of *justified* rows only (max 4 calls per run).
It can't change a verdict or cited note, and the rewrite is discarded if it drops the note id or adds any other note id or number (tested with a fake model in the harness).
**I have not run this path against a real Ollama server**, so its token numbers are not measured.

## Design choices and limits (things I would say in the walkthrough)

- **Why TF-IDF, not embeddings or a vector DB?** There are only 10 short notes and the route names are literally in the text, so TF-IDF retrieves well. I'd swap in sentence embeddings + Chroma if there were thousands of notes; `NoteIndex.retrieve` is the only place that would change.
- **Why not let an LLM decide the verdict?** The brief says a confident wrong reason is worse than "couldn't find one". Rules are easy to audit and repeat exactly; the LLM can only polish wording.
- **Note interpretation uses keyword cues** (`RISE_CUES`, `NO_EFFECT_CUES` in `notes.py`). It works for these 10 notes but a differently worded note (e.g. "costs were unaffected") would need a new cue. An LLM-based classifier would be the next step, with the same hard checks after it.
- **The 8-week window for level-change notes** is my interpretation, not in the brief.
- **A note with no stated magnitude is accepted** whatever the size of the rise (e.g. diesel says 5-7%, Mumbai-Delhi showed +11.6% and was still accepted). I did not add a magnitude check because I would be tuning it by eye.
- **Only rises are in the output.** Weeks that are normal are not listed; all weeks are in `weekly_metrics.csv`.
- `sample_output_format.csv` has an unquoted comma in its 3rd row's `reason`. My output quotes such fields properly so it parses cleanly.

## Folder layout

```
run.py                  entry point
repro_check.py          3-run comparison
costwatch/              weekly.py, flagging.py, notes.py, explain.py, qa.py
eval/                   eval_harness.py, labeled_set.csv
data/                   the three input files
outputs/                flagged_output.csv  <- the submission file
                        weekly_metrics.csv, token_cost_log.json, repro_log.txt
```
