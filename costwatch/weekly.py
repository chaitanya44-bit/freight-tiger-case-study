"""Step 1: shipments -> Monday-Sunday weeks -> cost per tonne-km -> the two baselines."""
import pandas as pd

HISTORY_WEEKS = 8  # trailing window for "vs own history" (fixed by the brief)


def load_shipments(path):
    df = pd.read_csv(path, parse_dates=["shipment_date"])
    # basic hygiene: a row with zero/negative/missing numbers would break cost per tonne-km
    num = ["quantity_tonnes", "distance_km", "freight_cost_inr"]
    bad = df[num].le(0).any(axis=1) | df[num + ["shipment_date"]].isna().any(axis=1)
    if bad.any():
        print(f"[warn] dropping {int(bad.sum())} unusable shipment rows")
        df = df[~bad]
    df = df.copy()
    df["route"] = df["origin"].str.strip() + "-" + df["destination"].str.strip()
    # week_of = Monday of the week the shipment falls in
    df["week_of"] = df["shipment_date"] - pd.to_timedelta(df["shipment_date"].dt.weekday, unit="D")
    return df


def weekly_table(df):
    """One row per route per week. Cost per tonne-km = sum(cost) / sum(tonnes * km)."""
    df = df.assign(tonne_km=df["quantity_tonnes"] * df["distance_km"])
    wk = df.groupby(["route", "route_type", "week_of"], as_index=False).agg(
        total_cost=("freight_cost_inr", "sum"),
        tonne_km=("tonne_km", "sum"),
        shipments=("shipment_id", "count"),
    )
    wk["cost_per_tonne_km"] = wk["total_cost"] / wk["tonne_km"]
    return wk.sort_values(["route", "week_of"]).reset_index(drop=True)


def add_baselines(wk):
    """Adds the two comparison numbers exactly as the brief defines them.

    own_avg  : mean of this route's weekly values over the 8 calendar weeks strictly
               before this week (fewer if the route has fewer prior weeks - no padding).
    peer_avg : mean of the other routes' value in the SAME week, same route_type.
    """
    own_avg, n_prior, peer_avg = [], [], []
    for r in wk.itertuples():
        window_start = r.week_of - pd.Timedelta(weeks=HISTORY_WEEKS)
        prior = wk[(wk.route == r.route) & (wk.week_of >= window_start) & (wk.week_of < r.week_of)]
        peers = wk[(wk.route_type == r.route_type) & (wk.week_of == r.week_of) & (wk.route != r.route)]
        own_avg.append(prior["cost_per_tonne_km"].mean() if len(prior) else float("nan"))
        n_prior.append(len(prior))
        peer_avg.append(peers["cost_per_tonne_km"].mean() if len(peers) else float("nan"))
    out = wk.copy()
    out["own_avg"], out["n_prior_weeks"], out["peer_avg"] = own_avg, n_prior, peer_avg
    out["own_pct"] = out["cost_per_tonne_km"] / out["own_avg"] - 1
    out["peer_pct"] = out["cost_per_tonne_km"] / out["peer_avg"] - 1
    return out
