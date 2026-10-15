"""Step 2: decide which route-weeks show a cost rise that looks out of the ordinary."""

# Thresholds were picked from the quiet part of the data (Jan-Oct 2024): there, the
# week-to-week wobble vs own history never went above ~8.5% and a route's normal gap
# to its peers stayed under ~17%. So +10% / +20% sits just above ordinary noise.
OWN_THRESHOLD = 0.10
PEER_THRESHOLD = 0.20
MIN_PRIOR_WEEKS = 4   # with <4 weeks of history the "average" is too noisy to flag on


def detect_rises(wk):
    """A row is a rise if cost is above its own history AND (far above own history OR far above peers)."""
    out = wk.copy()
    enough_history = out["n_prior_weeks"] >= MIN_PRIOR_WEEKS
    rising = out["own_pct"] > 0
    out["via_own"] = enough_history & rising & (out["own_pct"] >= OWN_THRESHOLD)
    out["via_peer"] = enough_history & rising & (out["peer_pct"] >= PEER_THRESHOLD)
    out["is_rise"] = out["via_own"] | out["via_peer"]
    return out
