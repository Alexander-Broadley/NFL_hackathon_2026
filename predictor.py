#!/usr/bin/env python3
"""
XGBoost player-trajectory predictor for the NFL Big Data Bowl tracking data.

Primary model (framing "B"): learn the per-frame DISPLACEMENT (dx, dy) a player
makes over one 0.1 s frame, as a function of that player's current kinematics
and relational context (ball + nearest opponent). At inference we start from the
snap frame and roll the model forward AUTOREGRESSIVELY, reconstructing absolute
(x, y) at every frame by accumulating predicted displacements.

Why displacement, not absolute (x, y)? Predicting absolute field position lets a
tree just memorise where plays happen. One-frame displacement is a near-linear,
physical quantity (position barely changes in 0.1 s) that generalises far better.

Optional: pass --per-horizon to train separate models for several
frames-since-snap bins (the "different frame transitions" idea) and compare.

Evaluation holds out WHOLE GAMES (grouped split) so no frame from a test play
ever appears in training. We report RMSE (yards) and mean Euclidean distance
error, broken down by frames-since-snap, against a constant-velocity baseline.

Usage:
    python predictor.py --games 30 --test-frac 0.2          # quick run
    python predictor.py --games 0                            # all 122 games
    python predictor.py --per-horizon                        # + horizon models
Outputs land in ./model_output/.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import xgboost as xgb
except Exception as e:  # pragma: no cover
    sys.exit(f"Failed to import xgboost ({e}). On macOS: conda install -n NFL_env "
             "-c conda-forge llvm-openmp")
from sklearn.model_selection import GroupShuffleSplit

# --------------------------------------------------------------------------- #
# Paths and constants
# --------------------------------------------------------------------------- #
HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
TRACKING_DIR = DATA / "tracking"
PLAYS_CSV = DATA / "plays.csv"
PLAYERS_CSV = DATA / "players.csv"
OUT_DIR = HERE / "model_output"

BALL_LABEL = "football"
SNAP_EVENTS = ("ball_snap", "autoevent_ballsnap")
DT = 0.1  # seconds per tracking frame (10 Hz)

# position groups collapsed to a small set of one-hot roles
POSITION_GROUPS = {
    "QB": "QB",
    "RB": "RB", "FB": "RB",
    "WR": "WR", "TE": "TE",
    "T": "OL", "G": "OL", "C": "OL",
    "DE": "DL", "DT": "DL", "NT": "DL",
    "OLB": "LB", "ILB": "LB", "MLB": "LB", "LB": "LB",
    "CB": "DB", "FS": "DB", "SS": "DB", "DB": "DB",
}
ROLE_ORDER = ["QB", "RB", "WR", "TE", "OL", "DL", "LB", "DB", "OTHER"]

FEATURE_COLS: list[str] = []  # filled by build_feature_frame (stable order)


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def list_game_ids(limit: int = 0) -> list[str]:
    ids = [f.stem.replace("tracking_", "")
           for f in sorted(TRACKING_DIR.glob("tracking_*.csv"))]
    return ids[:limit] if limit else ids


def load_players() -> pd.DataFrame:
    p = pd.read_csv(PLAYERS_CSV, usecols=["nflId", "officialPosition"])
    p["role"] = p["officialPosition"].map(POSITION_GROUPS).fillna("OTHER")
    return p[["nflId", "role"]]


def load_plays() -> pd.DataFrame:
    return pd.read_csv(PLAYS_CSV,
                       usecols=["gameId", "playId", "possessionTeam", "defensiveTeam"])


def load_tracking(game_id: str) -> pd.DataFrame:
    return pd.read_csv(TRACKING_DIR / f"tracking_{game_id}.csv")


# --------------------------------------------------------------------------- #
# Feature engineering
# --------------------------------------------------------------------------- #
def _snap_frame(play: pd.DataFrame) -> int:
    """frameId of the snap for a play (first matching snap event; else frame 1)."""
    for ev in SNAP_EVENTS:
        hit = play.loc[play["event"] == ev, "frameId"]
        if not hit.empty:
            return int(hit.min())
    return int(play["frameId"].min())


def _angles_to_components(speed, deg):
    """NGS angle (deg, 0 = +y, clockwise) -> (vx, vy) scaled by `speed`."""
    rad = np.deg2rad(deg)
    vx = speed * np.sin(rad)
    vy = speed * np.cos(rad)
    return vx, vy


def build_play_table(play: pd.DataFrame, players: pd.DataFrame,
                     poss_team: str) -> pd.DataFrame | None:
    """Return a per-player-per-frame table (snap frame onward) with features and
    next-frame displacement targets. Ball handled as its own 'role'."""
    snap = _snap_frame(play)
    play = play[play["frameId"] >= snap].copy()
    if play["frameId"].nunique() < 2:
        return None

    # normalise so the offense always moves in +x: flip left-moving plays.
    left = (play["playDirection"] == "left")
    play["x"] = np.where(left, 120.0 - play["x"], play["x"])
    play["y"] = np.where(left, 53.3 - play["y"], play["y"])
    # flip motion/orientation angles too (reflect about the y-axis of travel)
    for col in ("dir", "o"):
        play[col] = np.where(left, (360.0 - play[col]) % 360.0, play[col])

    play["is_ball"] = play["team"] == BALL_LABEL
    play["is_offense"] = play["team"] == poss_team
    play = play.merge(players, on="nflId", how="left")
    play.loc[play["is_ball"], "role"] = "BALL"
    play["role"] = play["role"].fillna("OTHER")

    play["s"] = play["s"].fillna(0.0)
    play["a"] = play["a"].fillna(0.0)
    play["dir"] = play["dir"].fillna(0.0)
    play["o"] = play["o"].fillna(0.0)

    vx, vy = _angles_to_components(play["s"].to_numpy(), play["dir"].to_numpy())
    play["vx"], play["vy"] = vx, vy

    # frames since snap, and snap-anchor position per entity
    play["fss"] = play["frameId"] - snap
    anchor = (play[play["fss"] == 0]
              .set_index(_entity_key(play[play["fss"] == 0]))[["x", "y"]]
              .rename(columns={"x": "ax", "y": "ay"}))
    play["ekey"] = _entity_key(play)
    play = play.join(anchor, on="ekey")
    # entities with no snap-frame row (rare) are dropped
    play = play.dropna(subset=["ax", "ay"]).copy()
    play["rx"] = play["x"] - play["ax"]   # position relative to snap anchor
    play["ry"] = play["y"] - play["ay"]

    # relational features computed per frame
    play = _add_relational(play)

    # Target = RESIDUAL displacement over a constant-velocity step. The model
    # only has to learn the deviation from physics (deceleration, turning,
    # reactions), not re-derive straight-line motion. At rollout we add the
    # physics term (v * dt) back. DT = 0.1 s per frame.
    play = play.sort_values(["ekey", "frameId"])
    full_dx = play.groupby("ekey")["x"].shift(-1) - play["x"]
    full_dy = play.groupby("ekey")["y"].shift(-1) - play["y"]
    play["tx"] = full_dx - play["vx"] * DT   # residual x
    play["ty"] = full_dy - play["vy"] * DT   # residual y

    return play


def _entity_key(df: pd.DataFrame) -> pd.Series:
    """Stable id per tracked entity within a play (ball gets a fixed key)."""
    return np.where(df["team"] == BALL_LABEL, "BALL",
                    df["nflId"].astype("Int64").astype(str))


def _add_relational(play: pd.DataFrame) -> pd.DataFrame:
    """For each frame add: distance+bearing to ball, and distance+bearing+speed
    to the nearest opponent. Vectorised per frame (<=23 entities each)."""
    out = []
    for fid, g in play.groupby("frameId", sort=False):
        g = g.copy()
        bx = g.loc[g["is_ball"], "x"]
        by = g.loc[g["is_ball"], "y"]
        bx = float(bx.iloc[0]) if not bx.empty else float(g["x"].mean())
        by = float(by.iloc[0]) if not by.empty else float(g["y"].mean())
        g["ball_dx"] = bx - g["x"]
        g["ball_dy"] = by - g["y"]
        g["ball_dist"] = np.hypot(g["ball_dx"], g["ball_dy"])

        # nearest opponent (players only; ball excluded)
        pl = g[~g["is_ball"]]
        px = pl["x"].to_numpy()
        py = pl["y"].to_numpy()
        off = pl["is_offense"].to_numpy()
        spd = pl["s"].to_numpy()
        n = len(pl)
        opp_dx = np.zeros(n); opp_dy = np.zeros(n)
        opp_dist = np.full(n, 20.0); opp_spd = np.zeros(n)
        for i in range(n):
            mask = off != off[i]
            if not mask.any():
                continue
            dx = px[mask] - px[i]
            dy = py[mask] - py[i]
            d = np.hypot(dx, dy)
            j = int(np.argmin(d))
            opp_dx[i], opp_dy[i], opp_dist[i] = dx[j], dy[j], d[j]
            opp_spd[i] = spd[mask][j]
        pl = pl.assign(opp_dx=opp_dx, opp_dy=opp_dy,
                       opp_dist=opp_dist, opp_spd=opp_spd)
        # ball rows get neutral opponent features
        gb = g[g["is_ball"]].assign(opp_dx=0.0, opp_dy=0.0,
                                    opp_dist=20.0, opp_spd=0.0)
        out.append(pd.concat([pl, gb], ignore_index=True))
    return pd.concat(out, ignore_index=True)


def feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Select/one-hot the model features in a stable column order."""
    base = df[["rx", "ry", "s", "a", "vx", "vy", "o", "dir", "fss",
               "is_offense", "ball_dx", "ball_dy", "ball_dist",
               "opp_dx", "opp_dy", "opp_dist", "opp_spd"]].copy()
    base["is_offense"] = base["is_offense"].astype(float)
    for r in ROLE_ORDER + ["BALL"]:
        base[f"role_{r}"] = (df["role"] == r).astype(float)
    global FEATURE_COLS
    FEATURE_COLS = list(base.columns)
    return base


# --------------------------------------------------------------------------- #
# Dataset assembly
# --------------------------------------------------------------------------- #
def assemble(game_ids: list[str]) -> pd.DataFrame:
    players = load_players()
    plays = load_plays()
    poss = {(int(r.gameId), int(r.playId)): r.possessionTeam
            for r in plays.itertuples(index=False)}
    frames = []
    for gi, gid in enumerate(game_ids, 1):
        tr = load_tracking(gid)
        for pid, play in tr.groupby("playId"):
            pteam = poss.get((int(gid), int(pid)))
            if pteam is None:
                continue
            tbl = build_play_table(play, players, pteam)
            if tbl is None:
                continue
            tbl["gameId"] = int(gid)
            tbl["playId"] = int(pid)
            frames.append(tbl)
        if gi % 10 == 0 or gi == len(game_ids):
            print(f"  assembled {gi}/{len(game_ids)} games", flush=True)
    data = pd.concat(frames, ignore_index=True)
    # drop final frame of each entity (no next-frame target)
    data = data.dropna(subset=["tx", "ty"]).reset_index(drop=True)
    return data


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def make_model() -> xgb.XGBRegressor:
    return xgb.XGBRegressor(
        n_estimators=400, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
        objective="reg:squarederror", n_jobs=-1, tree_method="hist",
    )


def train_models(train: pd.DataFrame, per_horizon: bool):
    """Return a dict describing trained models. Always trains a single global
    (dx, dy) pair; optionally also per-horizon pairs keyed by fss-bin."""
    Xtr = feature_matrix(train)
    models = {"global": {}, "per_horizon": {}, "horizon_bins": None}

    print("  training global dx/dy models ...", flush=True)
    mx, my = make_model(), make_model()
    mx.fit(Xtr, train["tx"]); my.fit(Xtr, train["ty"])
    models["global"] = {"dx": mx, "dy": my}

    if per_horizon:
        bins = [(0, 5), (5, 10), (10, 20), (20, 10_000)]
        models["horizon_bins"] = bins
        for lo, hi in bins:
            m = (train["fss"] >= lo) & (train["fss"] < hi)
            if m.sum() < 500:
                continue
            print(f"  training horizon model fss[{lo},{hi}) n={int(m.sum())}", flush=True)
            hx, hy = make_model(), make_model()
            hx.fit(Xtr[m.values], train.loc[m, "tx"])
            hy.fit(Xtr[m.values], train.loc[m, "ty"])
            models["per_horizon"][(lo, hi)] = {"dx": hx, "dy": hy}
    return models


def _pick_horizon_model(models, fss: int):
    if not models["per_horizon"]:
        return models["global"]
    for (lo, hi), m in models["per_horizon"].items():
        if lo <= fss < hi:
            return m
    return models["global"]


# --------------------------------------------------------------------------- #
# Autoregressive rollout + baseline (per play, on held-out data)
# --------------------------------------------------------------------------- #
def rollout_play(play_tbl: pd.DataFrame, models, use_horizon: bool) -> pd.DataFrame:
    """Predict (x, y) for every entity at every frame from the snap, rolling the
    displacement model forward. Returns a tidy frame with pred/true/baseline."""
    play_tbl = play_tbl.sort_values(["frameId", "ekey"]).copy()
    frames = sorted(play_tbl["frameId"].unique())
    snap = frames[0]

    # snap-frame state per entity
    snap_rows = play_tbl[play_tbl["frameId"] == snap].set_index("ekey")
    state = snap_rows[["x", "y", "s", "a", "vx", "vy", "o", "dir",
                       "is_offense", "is_ball", "role", "ax", "ay"]].copy()
    ekeys = list(state.index)

    # constant-velocity baseline uses snap velocity, held constant (0.1 s/frame)
    base_x = state["x"].copy(); base_y = state["y"].copy()

    truth = {ek: play_tbl[play_tbl["ekey"] == ek].set_index("frameId")[["x", "y"]]
             for ek in ekeys}

    records = []
    cur_x = state["x"].to_dict(); cur_y = state["y"].to_dict()
    # evolving kinematic state (updated each step from the predicted displacement)
    cur_vx = state["vx"].to_dict(); cur_vy = state["vy"].to_dict()
    cur_s = state["s"].to_dict(); cur_dir = state["dir"].to_dict()
    for ek in ekeys:
        records.append(dict(ekey=ek, frameId=snap, fss=0,
                            pred_x=cur_x[ek], pred_y=cur_y[ek],
                            base_x=base_x[ek], base_y=base_y[ek],
                            true_x=cur_x[ek], true_y=cur_y[ek],
                            role=state.loc[ek, "role"]))

    for step, fid in enumerate(frames[1:], start=1):
        fss_prev = fid - 1 - snap
        # build feature rows from CURRENT (evolving) predicted state
        rows = []
        for ek in ekeys:
            s = state.loc[ek]
            rows.append(dict(ekey=ek, x=cur_x[ek], y=cur_y[ek],
                             s=cur_s[ek], a=s["a"], vx=cur_vx[ek], vy=cur_vy[ek],
                             o=s["o"], dir=cur_dir[ek], is_offense=s["is_offense"],
                             is_ball=s["is_ball"], role=s["role"],
                             ax=s["ax"], ay=s["ay"], fss=fss_prev))
        fr = pd.DataFrame(rows)
        fr["frameId"] = fid  # single frame; needed by _add_relational's groupby
        fr["rx"] = fr["x"] - fr["ax"]; fr["ry"] = fr["y"] - fr["ay"]
        fr = _add_relational(fr)
        X = feature_matrix(fr)

        m = _pick_horizon_model(models, fss_prev) if use_horizon else models["global"]
        res_x = m["dx"].predict(X); res_y = m["dy"].predict(X)
        fr = fr.reset_index(drop=True)
        for i, ek in enumerate(fr["ekey"]):
            # full displacement = constant-velocity physics term + learned residual
            ddx = cur_vx[ek] * DT + float(res_x[i])
            ddy = cur_vy[ek] * DT + float(res_y[i])
            cur_x[ek] += ddx
            cur_y[ek] += ddy
            # update velocity from the predicted step, with light smoothing so
            # one noisy prediction can't whip the heading around.
            new_vx, new_vy = ddx / DT, ddy / DT
            cur_vx[ek] = 0.5 * cur_vx[ek] + 0.5 * new_vx
            cur_vy[ek] = 0.5 * cur_vy[ek] + 0.5 * new_vy
            cur_s[ek] = float(np.hypot(cur_vx[ek], cur_vy[ek]))
            if cur_s[ek] > 1e-6:
                cur_dir[ek] = float(np.rad2deg(np.arctan2(cur_vx[ek], cur_vy[ek])) % 360.0)

        # advance baseline by snap velocity * DT (constant-velocity extrapolation)
        for ek in ekeys:
            base_x[ek] += state.loc[ek, "vx"] * DT
            base_y[ek] += state.loc[ek, "vy"] * DT

        for ek in ekeys:
            t = truth[ek]
            tx = float(t.loc[fid, "x"]) if fid in t.index else np.nan
            ty = float(t.loc[fid, "y"]) if fid in t.index else np.nan
            records.append(dict(ekey=ek, frameId=fid, fss=fid - snap,
                                pred_x=cur_x[ek], pred_y=cur_y[ek],
                                base_x=base_x[ek], base_y=base_y[ek],
                                true_x=tx, true_y=ty,
                                role=state.loc[ek, "role"]))
    return pd.DataFrame(records)


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
def evaluate(test: pd.DataFrame, models, use_horizon: bool, max_plays: int = 0):
    play_keys = test[["gameId", "playId"]].drop_duplicates()
    if max_plays:
        play_keys = play_keys.head(max_plays)
    all_rows = []
    for i, (gid, pid) in enumerate(play_keys.itertuples(index=False), 1):
        tbl = test[(test["gameId"] == gid) & (test["playId"] == pid)]
        res = rollout_play(tbl, models, use_horizon)
        res = res[res["ekey"] != "BALL"]        # score players only
        res = res.dropna(subset=["true_x", "true_y"])
        all_rows.append(res)
        if i % 50 == 0 or i == len(play_keys):
            print(f"  rolled out {i}/{len(play_keys)} test plays", flush=True)
    r = pd.concat(all_rows, ignore_index=True)

    r["pred_err"] = np.hypot(r["pred_x"] - r["true_x"], r["pred_y"] - r["true_y"])
    r["base_err"] = np.hypot(r["base_x"] - r["true_x"], r["base_y"] - r["true_y"])

    def _rmse(a):
        return float(np.sqrt(np.mean(a ** 2)))

    overall = {
        "n_player_frames": int(len(r)),
        "model_mean_dist_err_yd": float(r["pred_err"].mean()),
        "model_rmse_dist_yd": _rmse(r["pred_err"]),
        "baseline_mean_dist_err_yd": float(r["base_err"].mean()),
        "baseline_rmse_dist_yd": _rmse(r["base_err"]),
    }
    by_h = (r.groupby("fss")
              .agg(n=("pred_err", "size"),
                   model_mean_err=("pred_err", "mean"),
                   baseline_mean_err=("base_err", "mean"))
              .reset_index())
    return overall, by_h, r


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--games", type=int, default=30,
                    help="number of games to load (0 = all 122)")
    ap.add_argument("--test-frac", type=float, default=0.2,
                    help="fraction of GAMES held out for testing")
    ap.add_argument("--per-horizon", action="store_true",
                    help="also train per-horizon models and use them in rollout")
    ap.add_argument("--eval-plays", type=int, default=0,
                    help="cap number of test plays rolled out (0 = all)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    OUT_DIR.mkdir(exist_ok=True)
    t0 = time.time()

    game_ids = list_game_ids(args.games)
    print(f"Loading {len(game_ids)} games ...", flush=True)
    data = assemble(game_ids)
    print(f"Assembled {len(data):,} player/ball-frame rows "
          f"across {data[['gameId','playId']].drop_duplicates().shape[0]:,} plays.",
          flush=True)

    # grouped split by GAME so no test play leaks into training
    gss = GroupShuffleSplit(n_splits=1, test_size=args.test_frac,
                            random_state=args.seed)
    tr_idx, te_idx = next(gss.split(data, groups=data["gameId"]))
    train, test = data.iloc[tr_idx].copy(), data.iloc[te_idx].copy()
    tr_games = sorted(train["gameId"].unique().tolist())
    te_games = sorted(test["gameId"].unique().tolist())
    print(f"Split by game: {len(tr_games)} train games, {len(te_games)} test games "
          f"({len(train):,} / {len(test):,} rows).", flush=True)

    models = train_models(train, args.per_horizon)

    print("Evaluating on held-out games (autoregressive rollout) ...", flush=True)
    overall, by_h, _ = evaluate(test, models, args.per_horizon, args.eval_plays)

    print("\n=== Overall (held-out games) ===")
    for k, v in overall.items():
        print(f"  {k}: {v}")
    improvement = (1 - overall["model_mean_dist_err_yd"]
                   / overall["baseline_mean_dist_err_yd"]) * 100
    print(f"  model beats constant-velocity baseline by {improvement:.1f}% "
          f"(mean distance error)")

    # persist
    overall["train_games"] = tr_games
    overall["test_games"] = te_games
    overall["per_horizon"] = args.per_horizon
    overall["improvement_vs_baseline_pct"] = improvement
    (OUT_DIR / "metrics.json").write_text(json.dumps(overall, indent=2))
    by_h.to_csv(OUT_DIR / "error_by_horizon.csv", index=False)
    models["global"]["dx"].save_model(OUT_DIR / "model_dx.json")
    models["global"]["dy"].save_model(OUT_DIR / "model_dy.json")
    (OUT_DIR / "feature_cols.json").write_text(json.dumps(FEATURE_COLS, indent=2))

    print(f"\nSaved metrics, per-horizon errors, and models to {OUT_DIR}")
    print(f"Done in {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
