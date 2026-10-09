#!/usr/bin/env python3
"""
NFL Big Data Bowl play visualiser.

A small Dash web app that animates player tracking data from
./data/tracking/ over all frames of a
chosen play. Players are drawn as coloured markers by team (one colour per
team on the play); the ball is a brown diamond. Jersey numbers are printed on
each player, and a short tick shows each player's orientation.

Run:
    python visualiser.py
then open the URL it prints (default http://127.0.0.1:8050).

Dependencies: dash, plotly, pandas. If any are missing the app prints the
exact conda command to install them into the active environment and exits.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# ---- dependency check with a friendly message -----------------------------
_missing = []
try:
    import pandas as pd
except ImportError:
    _missing.append("pandas")
try:
    import plotly.graph_objects as go
except ImportError:
    _missing.append("plotly")
try:
    import numpy as np
except ImportError:
    _missing.append("numpy")
try:
    from dash import Dash, dcc, html, Input, Output
except ImportError:
    _missing.append("dash")

if _missing:
    pkgs = " ".join(sorted(set(_missing)))
    sys.exit(
        "Missing required package(s): " + pkgs + "\n"
        "Install into the active conda env, e.g.:\n"
        f"    conda install -n NFL_env -c conda-forge {pkgs}\n"
        "  or:\n"
        f"    pip install {pkgs}\n"
    )

# ---- paths and field constants --------------------------------------------
DATA = Path(__file__).resolve().parent / "data"
TRACKING_DIR = DATA / "tracking"
PLAYS_CSV = DATA / "plays.csv"
GAMES_CSV = DATA / "games.csv"

FIELD_LENGTH = 120.0   # yards, including the two 10-yard end zones
FIELD_WIDTH = 53.3     # yards (field is 160 ft wide)
HASH_FROM_SIDELINE = 23.36667  # NFL hash marks ~23 yd 4 in from each sideline

BALL_LABEL = "football"
# teams drawn in the two NFL shield colours (blue vs red)
TEAM_COLORS = ["#013369", "#D50A0A"]  # team A (shield blue), team B (shield red)
BALL_COLOR = "#8B4513"                # brown football

# ---- NFL Big Data Bowl brand palette ---------------------------------------
# Anchored on the official NFL shield colours: blue #013369, red #D50A0A, white.
NFL_BLUE = "#013369"       # NFL Shield Blue (primary brand colour)
NFL_RED = "#D50A0A"        # NFL Shield Red (accent brand colour)

BG_PAGE = "#f4f6f8"        # light page background
BG_SURFACE = "#ffffff"     # card / island surface (white)
BG_SURFACE_2 = "#ffffff"   # inputs (white, for black readable text)
BORDER = "#d4dae2"         # hairline borders
TEXT = "#0b1220"           # primary text (near-black, high contrast)
TEXT_MUTED = "#5b6775"     # secondary text
ACCENT = NFL_RED           # accent (NFL red)

FIELD_GREEN = "#1a6b3c"    # classic turf green
FIELD_ENDZONE = "#013369"  # end zones in NFL shield blue
FIELD_LINE = "#ffffff"     # white field lines (standard NFL markings)


# ---- data loading ----------------------------------------------------------
def list_game_ids() -> list[str]:
    """gameIds for which a tracking file exists, as strings, sorted."""
    ids = []
    for f in sorted(TRACKING_DIR.glob("tracking_*.csv")):
        gid = f.stem.replace("tracking_", "")
        ids.append(gid)
    return ids


def tracking_path(game_id: str) -> Path:
    return TRACKING_DIR / f"tracking_{game_id}.csv"


def load_tracking(game_id: str) -> pd.DataFrame:
    """Load one game's tracking file. Cached per-process to avoid re-reading."""
    df = _TRACKING_CACHE.get(game_id)
    if df is None:
        df = pd.read_csv(tracking_path(game_id))
        _TRACKING_CACHE[game_id] = df
    return df


_TRACKING_CACHE: dict[str, pd.DataFrame] = {}


def load_plays() -> pd.DataFrame:
    if PLAYS_CSV.exists():
        cols = ["gameId", "playId", "playDescription", "quarter",
                "gameClock", "possessionTeam", "defensiveTeam",
                "playResult", "prePenaltyPlayResult", "passResult",
                "absoluteYardlineNumber"]
        try:
            return pd.read_csv(PLAYS_CSV, usecols=cols)
        except ValueError:
            return pd.read_csv(PLAYS_CSV)
    return pd.DataFrame()


PLAYS = load_plays()
GAME_IDS = list_game_ids()


# ---- prediction bridge ------------------------------------------------------
# Reuse the trained XGBoost displacement models + rollout from predictor.py so
# the overlay always matches the real model (no duplicated feature logic).
REPO_DIR = Path(__file__).resolve().parent
MODEL_DIR = REPO_DIR / "model_output"
METRICS_JSON = MODEL_DIR / "metrics.json"

_PRED = {"ok": False, "reason": "", "predictor": None, "models": None,
         "players": None, "poss": None}
_PRED_CACHE: dict[tuple, pd.DataFrame] = {}

# game-level train/test split read from the model's metrics.json. A game can be
# a TRAIN game (model learned from it), a TEST game (held out, truly unseen), or
# UNSEEN (not part of the model's run at all - also never trained on, but not
# the designated evaluation set either).
_TRAIN_GAMES: set[str] = set()
_TEST_GAMES: set[str] = set()


def _load_split() -> None:
    """Populate _TRAIN_GAMES / _TEST_GAMES from metrics.json (best effort)."""
    if _TRAIN_GAMES or _TEST_GAMES:
        return
    if not METRICS_JSON.exists():
        return
    try:
        m = json.loads(METRICS_JSON.read_text())
        _TRAIN_GAMES.update(str(g) for g in m.get("train_games", []))
        _TEST_GAMES.update(str(g) for g in m.get("test_games", []))
    except Exception:
        pass


def game_split_label(game_id) -> str:
    """'train', 'test', or 'unseen' for a given gameId."""
    _load_split()
    gid = str(game_id)
    if gid in _TEST_GAMES:
        return "test"
    if gid in _TRAIN_GAMES:
        return "train"
    return "unseen"


# display metadata per split state: (short label, long tooltip, colour)
SPLIT_META = {
    "train": ("TRAIN", "Model was trained on this game — predictions here are "
                        "in-sample and will look optimistically good.",
              "#fbbf24"),   # amber
    "test":  ("TEST", "Held-out game the model never saw during training — "
                      "this is an honest, out-of-sample prediction.",
              "#4ade80"),   # green
    "unseen": ("UNSEEN", "Not part of the model's train/test run. The model "
                         "never trained on it, but it isn't the designated "
                         "test set either.",
               "#8b98a9"),  # grey
}


def _init_predictor() -> None:
    """Lazy, best-effort load of predictor module + saved models. Failures are
    captured in _PRED['reason'] so the UI can degrade gracefully."""
    if _PRED["ok"] or _PRED["reason"]:
        return
    try:
        sys.path.insert(0, str(REPO_DIR))
        import predictor as _p  # noqa: E402
        import xgboost as _xgb  # noqa: E402

        dxp = MODEL_DIR / "model_dx.json"
        dyp = MODEL_DIR / "model_dy.json"
        if not (dxp.exists() and dyp.exists()):
            _PRED["reason"] = "no saved model (run predictor.py first)"
            return
        mx, my = _xgb.XGBRegressor(), _xgb.XGBRegressor()
        mx.load_model(dxp); my.load_model(dyp)
        _PRED["predictor"] = _p
        _PRED["models"] = {"global": {"dx": mx, "dy": my},
                           "per_horizon": {}, "horizon_bins": None}
        _PRED["players"] = _p.load_players()
        plays = _p.load_plays()
        _PRED["poss"] = {(int(r.gameId), int(r.playId)): r.possessionTeam
                         for r in plays.itertuples(index=False)}
        _PRED["ok"] = True
    except Exception as e:  # pragma: no cover - defensive
        _PRED["reason"] = f"predictor unavailable: {e}"


def predictions_available() -> bool:
    _init_predictor()
    return _PRED["ok"]


def predicted_positions(game_id: str, play_id: int) -> pd.DataFrame | None:
    """Rolled-out predicted positions for a play, in ORIGINAL field coordinates
    (un-flipped), keyed by nflId and frameId. None if unavailable.

    Columns: nflId (Int64), frameId (int), px (float), py (float).
    """
    _init_predictor()
    if not _PRED["ok"]:
        return None
    key = (str(game_id), int(play_id))
    if key in _PRED_CACHE:
        return _PRED_CACHE[key]

    p = _PRED["predictor"]
    pteam = _PRED["poss"].get((int(game_id), int(play_id)))
    if pteam is None:
        return None
    tr = load_tracking(game_id)
    play = tr[tr["playId"] == play_id]
    if play.empty:
        return None

    tbl = p.build_play_table(play, _PRED["players"], pteam)
    if tbl is None:
        return None
    roll = p.rollout_play(tbl, _PRED["models"], use_horizon=False)
    roll = roll[roll["ekey"] != "BALL"].copy()

    # predictor flips left-moving plays to +x; un-flip back to original coords.
    play_dir = play["playDirection"].iloc[0]
    px = roll["pred_x"].to_numpy()
    py = roll["pred_y"].to_numpy()
    if play_dir == "left":
        px = FIELD_LENGTH - px
        py = FIELD_WIDTH - py
    out = pd.DataFrame({
        "nflId": pd.to_numeric(roll["ekey"], errors="coerce").astype("Int64"),
        "frameId": roll["frameId"].astype(int).to_numpy(),
        "px": px, "py": py,
    }).dropna(subset=["nflId"])
    _PRED_CACHE[key] = out
    return out


def predicted_play_result(game_id: str, play_id: int) -> int | None:
    """Predicted net yards gained, derived geometrically from the LAST frame of
    the simulated (model-rolled-out) play. None when it can't be computed.

    The ball-carrier is approximated as the furthest-downfield offensive player
    at the final simulated frame. 'Downfield' is the direction the offense is
    driving, read off the play's playDirection. Yards = how far that player has
    advanced past the line of scrimmage (absoluteYardlineNumber).

    We work entirely in ORIGINAL (un-flipped) field coordinates: predicted_positions
    returns original coords, and absoluteYardlineNumber is in the same 0-120 space,
    so there is no normalised/original mixing. For a 'right' play the offense moves
    toward +x, so yards = final_x - LOS_x; for a 'left' play it moves toward -x, so
    yards = LOS_x - final_x.
    """
    if not predictions_available():
        return None
    pp = predicted_positions(game_id, play_id)
    if pp is None or pp.empty:
        return None

    info = play_info(game_id, play_id)
    los_x = info.get("absoluteYardlineNumber")
    if los_x is None:
        return None

    # possession team code -> which nflIds are on offense for this play
    pteam = _PRED["poss"].get((int(game_id), int(play_id)))
    if pteam is None:
        return None
    tr = load_tracking(game_id)
    play = tr[tr["playId"] == int(play_id)]
    if play.empty:
        return None
    play_dir = play["playDirection"].iloc[0]
    off_ids = set(
        int(n) for n in play[play["team"] == pteam]["nflId"].dropna().unique()
    )
    if not off_ids:
        return None

    # last simulated frame, offensive players only
    final_fid = int(pp["frameId"].max())
    final = pp[(pp["frameId"] == final_fid) & (pp["nflId"].isin(off_ids))]
    if final.empty:
        return None

    # furthest-downfield offensive player = ball-carrier approximation
    if play_dir == "left":
        carrier_x = float(final["px"].min())  # offense drives toward -x
        yards = los_x - carrier_x
    else:  # 'right' (default): offense drives toward +x
        carrier_x = float(final["px"].max())
        yards = carrier_x - los_x
    return int(round(yards))


def play_options(game_id: str) -> list[dict]:
    """Dropdown options for every play in a game, labelled from plays.csv."""
    df = load_tracking(game_id)
    play_ids = sorted(df["playId"].unique().tolist())
    opts = []
    gid_int = int(game_id)
    for pid in play_ids:
        label = f"Play {pid}"
        if not PLAYS.empty:
            match = PLAYS[(PLAYS["gameId"] == gid_int) & (PLAYS["playId"] == pid)]
            if not match.empty and "playDescription" in match:
                desc = str(match.iloc[0]["playDescription"])
                label = f"{pid}: {desc[:80]}"
        opts.append({"label": label, "value": pid})
    return opts


def play_info(game_id: str, play_id: int) -> dict:
    """Look up play-level info from plays.csv for a given gameId+playId.

    Returns a dict with:
      'playResult'            net yards gained incl. penalty yards (int or None)
      'prePenaltyPlayResult'  net yards before penalty yards (int or None)
      'passResult'            dropback outcome code C/I/S/IN/R (str or None)
      'playDescription'       play text (str or None)
      'absoluteYardlineNumber' absolute x of the LOS in 0-120 coords (int/None)
    """
    info = {"playResult": None, "prePenaltyPlayResult": None,
            "passResult": None, "playDescription": None,
            "absoluteYardlineNumber": None}
    if PLAYS.empty:
        return info
    match = PLAYS[(PLAYS["gameId"] == int(game_id)) & (PLAYS["playId"] == int(play_id))]
    if match.empty:
        return info
    row = match.iloc[0]
    for col in ("playResult", "prePenaltyPlayResult", "absoluteYardlineNumber"):
        if col in match.columns and pd.notna(row[col]):
            info[col] = int(row[col])
    if "passResult" in match.columns and pd.notna(row["passResult"]):
        info["passResult"] = str(row["passResult"])
    if "playDescription" in match.columns and pd.notna(row["playDescription"]):
        info["playDescription"] = str(row["playDescription"])
    return info


def _result_text(play_result) -> str:
    """Human-readable yardage outcome, e.g. '+8 yards', '-2 yards', 'no gain'."""
    if play_result is None:
        return "result: n/a"
    if play_result == 0:
        return "result: no gain"
    sign = "+" if play_result > 0 else ""
    return f"result: {sign}{play_result} yd"


# passResult codes -> (readable label, colour class)
PASS_RESULT_MAP = {
    "C": ("Complete", "result-gain"),
    "I": ("Incomplete", "result-zero"),
    "S": ("Sack", "result-loss"),
    "IN": ("Interception", "result-loss"),
    "R": ("Scramble", "result-gain"),
}


def _yardage_badge_parts(value):
    """(display string, css class) for a signed-yardage value or None."""
    if value is None:
        return "n/a", "result-zero"
    if value > 0:
        return f"+{value} yd", "result-gain"
    if value < 0:
        return f"{value} yd", "result-loss"
    return "0 yd", "result-zero"


def _passresult_badge_parts(code):
    """(display string, css class) for a passResult code or None."""
    if code is None:
        return "—", "result-zero"
    label, cls = PASS_RESULT_MAP.get(code, (code, "result-zero"))
    return label, cls


def _badge(label, value, cls):
    """One outcome badge: an uppercase label over a coloured value."""
    return html.Div(className="result-badge", children=[
        html.Span(label, className="result-label"),
        html.Span(value, className=f"result-value {cls}"),
    ])


def _badges_children(game_id, play_id, view_mode="actual"):
    """Children for the outcome-badges row: pass result + two yardage figures.

    When view_mode is 'pred' or 'both', two extra badges are appended that
    surface the model's predicted net yards (from the final simulated frame)
    and the signed delta against the actual play result (predicted minus
    actual; positive = model over-predicted the gain). Badges are omitted
    (never crash) when predictions or the inputs they need are unavailable.
    """
    if game_id is None or play_id is None:
        info = {"playResult": None, "prePenaltyPlayResult": None,
                "passResult": None}
    else:
        info = play_info(game_id, play_id)

    pass_val, pass_cls = _passresult_badge_parts(info["passResult"])
    pre_val, pre_cls = _yardage_badge_parts(info["prePenaltyPlayResult"])
    net_val, net_cls = _yardage_badge_parts(info["playResult"])
    children = [
        _badge("Pass result", pass_val, pass_cls),
        _badge("Pre-penalty", pre_val, pre_cls),
        _badge("Play result", net_val, net_cls),
    ]

    if view_mode in ("pred", "both") and game_id is not None and play_id is not None:
        try:
            pred = predicted_play_result(game_id, play_id)
        except Exception:  # pragma: no cover - defensive, never break the UI
            pred = None
        if pred is not None:
            pred_val, pred_cls = _yardage_badge_parts(pred)
            children.append(_badge("Predicted result", pred_val, pred_cls))
            actual = info["playResult"]
            if actual is not None:
                delta_val, delta_cls = _yardage_badge_parts(pred - actual)
            else:
                delta_val, delta_cls = _yardage_badge_parts(None)
            children.append(_badge("Pred vs actual", delta_val, delta_cls))
    return children


# ---- figure building --------------------------------------------------------
def field_shapes() -> list[dict]:
    """Static field markings: end zones, yard lines, hash marks."""
    shapes = []
    # playing field background
    shapes.append(dict(type="rect", x0=0, y0=0, x1=FIELD_LENGTH, y1=FIELD_WIDTH,
                        line=dict(width=0), fillcolor=FIELD_GREEN, layer="below"))
    # end zones (0-10 and 110-120)
    for x0, x1 in [(0, 10), (110, 120)]:
        shapes.append(dict(type="rect", x0=x0, y0=0, x1=x1, y1=FIELD_WIDTH,
                           line=dict(width=0), fillcolor=FIELD_ENDZONE, layer="below"))
    # yard lines every 5 yards between the goal lines
    for x in range(10, 115, 5):
        shapes.append(dict(type="line", x0=x, y0=0, x1=x, y1=FIELD_WIDTH,
                           line=dict(color=FIELD_LINE, width=1), layer="below"))
    # sidelines / goal lines border
    shapes.append(dict(type="rect", x0=10, y0=0, x1=110, y1=FIELD_WIDTH,
                       line=dict(color=FIELD_LINE, width=2), layer="below"))
    # hash marks every yard
    for x in range(11, 110):
        for y in (HASH_FROM_SIDELINE, FIELD_WIDTH - HASH_FROM_SIDELINE):
            shapes.append(dict(type="line", x0=x, y0=y - 0.3, x1=x, y1=y + 0.3,
                               line=dict(color=FIELD_LINE, width=1), layer="below"))
    return shapes


def yardline_annotations() -> list[dict]:
    """Yard-number labels (10..50..10) along the field, both directions."""
    anns = []
    # numbers at 20,30,...,100 map to 10,20,30,40,50,40,...,10
    for x in range(20, 101, 10):
        num = x - 10 if x <= 60 else 110 - x
        for y in (8, FIELD_WIDTH - 8):
            anns.append(dict(x=x, y=y, text=str(num), showarrow=False,
                             font=dict(color=FIELD_LINE, size=14)))
    return anns


def _orientation_segments(sub: pd.DataFrame, length: float = 1.8):
    """Return x/y line segments (with None breaks) for player orientation `o`.
    `o` is in degrees, 0 = facing +y (toward the far sideline in NGS convention).
    We draw a short tick in that heading."""
    xs, ys = [], []
    rad = np.deg2rad(sub["o"].to_numpy())
    # NGS: 0 deg points toward increasing y; angle increases clockwise.
    dx = length * np.sin(rad)
    dy = length * np.cos(rad)
    px = sub["x"].to_numpy()
    py = sub["y"].to_numpy()
    for i in range(len(sub)):
        xs.extend([px[i], px[i] + dx[i], None])
        ys.extend([py[i], py[i] + dy[i], None])
    return xs, ys


def build_figure(game_id: str, play_id: int, mode: str = "actual") -> go.Figure:
    """mode: 'actual' (tracking only), 'pred' (model's predicted play only),
    or 'both' (actual solid + predicted ghost overlay)."""
    df = load_tracking(game_id)
    play = df[df["playId"] == play_id].copy()
    if play.empty:
        return go.Figure()

    play.sort_values(["frameId", "nflId"], inplace=True)
    frames_ids = sorted(play["frameId"].unique().tolist())

    want_pred = mode in ("pred", "both")
    pred_only = mode == "pred"

    # predicted tracks: per-nflId/frame predicted x,y in original coords.
    # pred_lookup[(nflId, frameId)] -> (px, py); pred_team: nflId -> team code.
    pred_lookup: dict[tuple, tuple] = {}
    pred_team: dict[int, str] = {}
    pred_start_fid = None  # first frame for which predictions exist (the snap)
    if want_pred:
        pp = predicted_positions(game_id, play_id)
        if pp is not None and not pp.empty:
            for r in pp.itertuples(index=False):
                pred_lookup[(int(r.nflId), int(r.frameId))] = (r.px, r.py)
            pred_start_fid = int(pp["frameId"].min())
            tmap = (play[play["team"] != BALL_LABEL]
                    .dropna(subset=["nflId"])
                    .drop_duplicates("nflId")[["nflId", "team"]])
            pred_team = {int(n): t for n, t in
                         zip(tmap["nflId"], tmap["team"])}
    # if predicted mode was requested but nothing came back, fall back to actual
    if want_pred and not pred_lookup:
        want_pred = False
        pred_only = False

    # play outcome from plays.csv, matched on gameId+playId
    info = play_info(game_id, play_id)
    result_txt = _result_text(info["playResult"])
    if info["passResult"] is not None:
        pass_label = PASS_RESULT_MAP.get(info["passResult"],
                                         (info["passResult"], ""))[0]
        result_txt = f"{pass_label}  ·  {result_txt}"
    # make the active view + data split obvious in the title
    mode_tag = {"pred": "PREDICTED", "both": "ACTUAL + PREDICTED"}.get(mode, "ACTUAL")
    split_tag = SPLIT_META[game_split_label(game_id)][0]  # TRAIN / TEST / UNSEEN
    result_txt = f"[{mode_tag} · {split_tag} game]  ·  {result_txt}"

    # team ordering: football last; two real teams get stable colours
    teams = [t for t in play["team"].unique().tolist() if t != BALL_LABEL]
    teams = sorted(teams)
    color_for = {t: TEAM_COLORS[i % len(TEAM_COLORS)] for i, t in enumerate(teams)}
    color_for[BALL_LABEL] = BALL_COLOR

    def _pred_xy_for_team(t: str, fid: int):
        """Predicted (x list, y list, jersey list) for a team at a frame, in the
        same player order as the tracking rows (so jersey labels line up)."""
        sub = play[(play["frameId"] == fid) & (play["team"] == t)]
        xs, ys, nums = [], [], []
        for r in sub.itertuples(index=False):
            if pd.isna(r.nflId):
                continue
            pos = pred_lookup.get((int(r.nflId), fid))
            if pos is None:
                # before the snap there is no prediction yet; fall back to the
                # actual position so the play starts from the real formation.
                pos = (r.x, r.y)
            xs.append(pos[0]); ys.append(pos[1])
            nums.append(r.jerseyNumber)
        return xs, ys, nums

    def frame_traces(fid: int) -> list[go.Scatter]:
        f = play[play["frameId"] == fid]
        traces = []
        # primary solid markers per team. In 'pred' mode these ARE the model's
        # predicted positions; otherwise they are the actual tracking positions.
        for t in teams:
            sub = f[f["team"] == t]
            if pred_only:
                px, py, nums = _pred_xy_for_team(t, fid)
                num_text = pd.Series(nums).astype("Int64").astype(str)
                suffix = " (pred)"
            else:
                px, py = sub["x"], sub["y"]
                num_text = sub["jerseyNumber"].astype("Int64").astype(str)
                suffix = ""
            traces.append(go.Scatter(
                x=px, y=py, mode="markers+text",
                marker=dict(size=16, color=color_for[t],
                            line=dict(color="white", width=1)),
                text=num_text,
                textposition="middle center",
                textfont=dict(color="white", size=9),
                name=f"{t}{suffix}", legendgroup=t,
                hovertemplate=(f"{t}{suffix}<br>#%{{text}}<br>"
                               "x=%{x:.1f}, y=%{y:.1f}<extra></extra>"),
            ))
            # orientation ticks: only meaningful for actual positions
            if not pred_only and not sub.empty:
                ox, oy = _orientation_segments(sub)
                traces.append(go.Scatter(
                    x=ox, y=oy, mode="lines",
                    line=dict(color=color_for[t], width=1),
                    hoverinfo="skip", showlegend=False, legendgroup=t,
                ))
        # the ball (always from actual tracking; the model doesn't predict it)
        ball = f[f["team"] == BALL_LABEL]
        traces.append(go.Scatter(
            x=ball["x"], y=ball["y"], mode="markers",
            marker=dict(size=10, color=BALL_COLOR, symbol="diamond",
                        line=dict(color="white", width=1)),
            name="ball", hovertemplate="ball<br>x=%{x:.1f}, y=%{y:.1f}<extra></extra>",
        ))

        # 'both' mode only: predicted ghosts overlaid on the actual markers.
        if mode == "both" and pred_lookup:
            gx_by_team: dict[str, list] = {t: [] for t in teams}
            gy_by_team: dict[str, list] = {t: [] for t in teams}
            for nid, team in pred_team.items():
                pos = pred_lookup.get((nid, fid))
                if pos is None or team not in gx_by_team:
                    continue
                gx_by_team[team].append(pos[0])
                gy_by_team[team].append(pos[1])
            for t in teams:
                if not gx_by_team[t]:
                    continue
                traces.append(go.Scatter(
                    x=gx_by_team[t], y=gy_by_team[t], mode="markers",
                    marker=dict(size=15, symbol="circle-open",
                                color=color_for[t],
                                line=dict(color=color_for[t], width=2),
                                opacity=0.55),
                    name=f"{t} (pred)", legendgroup=f"{t}-pred",
                    hovertemplate=(f"{t} predicted<br>"
                                   "x=%{x:.1f}, y=%{y:.1f}<extra></extra>"),
                ))
        return traces

    # initial data = first frame
    init_traces = frame_traces(frames_ids[0])

    # animation frames retain the static yard-number annotations.
    static_anns = yardline_annotations()
    go_frames = []
    for fid in frames_ids:
        ev = play[play["frameId"] == fid]["event"].iloc[0]
        ev_txt = "" if ev in (None, "None", "nan") else f"  |  event: {ev}"
        go_frames.append(go.Frame(
            data=frame_traces(fid), name=str(fid),
            layout=go.Layout(
                title_text=_title(game_id, play_id, fid,
                                  len(frames_ids), ev_txt, result_txt),
                annotations=static_anns),
        ))

    fig = go.Figure(data=init_traces, frames=go_frames)

    # slider steps
    # map frameId -> tagged event (first non-null per frame), for slider marks
    event_for = {}
    for fid in frames_ids:
        evs = play[play["frameId"] == fid]["event"].dropna()
        evs = [e for e in evs.tolist() if e not in ("None", "nan", "")]
        event_for[fid] = evs[0] if evs else None

    def _step_label(fid: int) -> str:
        # mark "real" tagged events (ignore the autoevent_* duplicates) with a dot
        ev = event_for.get(fid)
        return "●" if (ev and not ev.startswith("autoevent_")) else str(fid)

    steps = [dict(method="animate", label=_step_label(fid),
                  args=[[str(fid)],
                        dict(mode="immediate",
                             frame=dict(duration=0, redraw=True),
                             transition=dict(duration=0))])
             for fid in frames_ids]

    # playback at several speeds. (Plotly's static animation buttons play once
    # through to the last frame; there is no native "loop" arg, so pressing a
    # speed button again replays from the current frame.)
    def _play_btn(label: str, duration: int) -> dict:
        return dict(label=label, method="animate",
                    args=[None, dict(frame=dict(duration=duration, redraw=True),
                                     fromcurrent=True, mode="immediate",
                                     transition=dict(duration=0))])

    speed_slow = _play_btn("▶ 0.5×", 300)
    speed_1x = _play_btn("▶ 1×", 150)
    speed_fast = _play_btn("▶ 2×", 75)
    pause_btn = dict(label="⏸ Pause", method="animate",
                     args=[[None], dict(frame=dict(duration=0, redraw=False),
                                        mode="immediate",
                                        transition=dict(duration=0))])

    fig.update_layout(
        title=dict(text=_title(game_id, play_id, frames_ids[0], len(frames_ids),
                               "", result_txt),
                   font=dict(color=TEXT, size=15), x=0.01, xanchor="left"),
        font=dict(color=TEXT, family="Inter, system-ui, sans-serif"),
        xaxis=dict(range=[0, FIELD_LENGTH], showgrid=False, zeroline=False,
                   visible=False, constrain="domain"),
        yaxis=dict(range=[0, FIELD_WIDTH], showgrid=False, zeroline=False,
                   visible=False, scaleanchor="x", scaleratio=1),
        shapes=field_shapes(),
        annotations=yardline_annotations(),
        plot_bgcolor=BG_SURFACE, paper_bgcolor=BG_SURFACE,
        height=560, margin=dict(l=10, r=10, t=64, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1, font=dict(color=TEXT),
                    bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(bgcolor=BG_SURFACE_2, bordercolor=BORDER,
                        font=dict(color=TEXT)),
        updatemenus=[dict(type="buttons", showactive=False, x=0.02, y=0,
                          xanchor="left", yanchor="top", direction="left",
                          pad=dict(t=70, r=10),
                          bgcolor=BG_SURFACE_2, bordercolor=BORDER,
                          font=dict(color=TEXT),
                          buttons=[speed_slow, speed_1x, speed_fast, pause_btn])],
        sliders=[dict(active=0, x=0.1, len=0.88, xanchor="left", y=0,
                      yanchor="top", pad=dict(t=50, b=10),
                      bgcolor=BG_SURFACE_2, bordercolor=BORDER,
                      activebgcolor=ACCENT, tickcolor=TEXT_MUTED,
                      font=dict(color=TEXT_MUTED, size=10),
                      currentvalue=dict(prefix="Frame: ",
                                        font=dict(color=TEXT, size=12)),
                      steps=steps)],
    )
    return fig


def _title(game_id, play_id, fid, nframes, extra, result="") -> str:
    res = f"  ·  {result}" if result else ""
    return (f"Game {game_id} — Play {play_id} — frame {fid}/{nframes}{res}{extra}"
            "<br><span style='font-size:11px;color:#9db6d8'>"
            "NFL Big Data Bowl · Next Gen Stats</span>")


# ---- Dash app ---------------------------------------------------------------
app = Dash(__name__)
app.title = "NFL Big Data Bowl — Play Visualiser"

# Global dark theme + card styling. Injected at the page level so the dropdown
# menus, scrollbars and body background are themed too, not just components.
app.index_string = """<!DOCTYPE html>
<html>
<head>
    {%metas%}
    <title>{%title%}</title>
    {%favicon%}
    {%css%}
    <style>
        :root {
            --nfl-blue: #013369;
            --nfl-red: #D50A0A;
            --bg-page: #f4f6f8;
            --bg-surface: #ffffff;
            --bg-surface-2: #ffffff;
            --border: #d4dae2;
            --text: #0b1220;
            --text-muted: #5b6775;
            --accent: #D50A0A;
        }
        * { box-sizing: border-box; }
        body {
            margin: 0;
            background: var(--bg-page);
            color: var(--text);
            font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
            -webkit-font-smoothing: antialiased;
        }
        .app-shell { max-width: 1120px; margin: 0 auto; padding: 32px 20px 48px; }

        /* NFL-branded header banner */
        .app-header {
            display: flex; align-items: center; gap: 16px; margin-bottom: 22px;
            background: linear-gradient(90deg, var(--nfl-blue) 0%, #012a56 100%);
            border-radius: 16px; padding: 20px 24px;
            box-shadow: 0 8px 24px rgba(1,51,105,0.25);
            border-bottom: 4px solid var(--nfl-red);
        }
        .app-shield {
            width: 44px; height: 56px; flex: 0 0 auto;
            display: inline-flex; align-items: center; justify-content: center;
            background: #ffffff; color: var(--nfl-blue);
            border-radius: 8px 8px 20px 20px; border: 2px solid var(--nfl-red);
            font-weight: 800; font-size: 11px; letter-spacing: 0.04em;
            line-height: 1; text-align: center;
        }
        .app-header-text { display: flex; flex-direction: column; gap: 4px; }
        .app-title {
            font-size: 23px; font-weight: 800; letter-spacing: -0.01em;
            margin: 0; color: #ffffff; text-transform: uppercase;
        }
        .app-title .accent { color: var(--nfl-red); }
        .app-subtitle { font-size: 13px; color: #c7d2e0; margin: 0; }
        .card {
            background: var(--bg-surface);
            border: 1px solid var(--border);
            border-radius: 16px;
            box-shadow: 0 6px 20px rgba(1,51,105,0.08);
        }
        .controls-card {
            padding: 18px 20px; margin-bottom: 20px;
            border-top: 3px solid var(--nfl-blue);
        }
        .controls-row { display: flex; gap: 18px; flex-wrap: wrap; align-items: flex-end; }
        .control { display: flex; flex-direction: column; gap: 6px; }
        .control-label {
            font-size: 11px; font-weight: 700; letter-spacing: 0.06em;
            text-transform: uppercase; color: var(--nfl-blue);
        }
        /* the pitch island */
        .pitch-card { padding: 14px 14px 6px; }
        .pitch-caption {
            display: flex; gap: 18px; flex-wrap: wrap;
            padding: 10px 6px 4px; font-size: 12px; color: var(--text-muted);
        }
        .legend-chip { display: inline-flex; align-items: center; gap: 7px; }
        .dot { width: 11px; height: 11px; border-radius: 50%; display: inline-block;
               border: 1px solid rgba(0,0,0,0.25); }
        .diamond { width: 10px; height: 10px; display: inline-block; transform: rotate(45deg);
                   border: 1px solid rgba(0,0,0,0.25); }

        /* Dash dropdowns (react-select): white background, BLACK text */
        .Select-control, .is-focused:not(.is-open) > .Select-control {
            background: #ffffff !important;
            border: 1px solid var(--border) !important;
            border-radius: 10px !important;
            color: #000000 !important;
            box-shadow: none !important;
        }
        .Select-menu-outer {
            background: #ffffff !important;
            border: 1px solid var(--border) !important;
            border-radius: 10px !important;
            color: #000000 !important;
            overflow: hidden;
        }
        .Select-value-label, .Select-placeholder, .Select input > input,
        .Select-value { color: #000000 !important; }
        .VirtualizedSelectOption, .Select-option {
            background: #ffffff !important;
            color: #000000 !important;
        }
        .VirtualizedSelectFocusedOption, .Select-option.is-focused {
            background: #e8eef6 !important;
            color: #000000 !important;
        }
        .Select-arrow { border-color: var(--text-muted) transparent transparent; }

        /* play-outcome badges */
        .result-badges { display: flex; gap: 10px; flex-wrap: wrap; }
        .result-badge {
            display: inline-flex; flex-direction: column; gap: 2px;
            padding: 8px 16px; border-radius: 10px;
            background: #f7f9fc; border: 1px solid var(--border);
            min-width: 120px;
        }
        .result-label {
            font-size: 10px; font-weight: 700; letter-spacing: 0.06em;
            text-transform: uppercase; color: var(--nfl-blue);
        }
        .result-value { font-size: 20px; font-weight: 800; line-height: 1.1;
                        color: var(--text); }
        .result-gain { color: #1a8f3c; }   /* green for positive yards */
        .result-loss { color: var(--nfl-red); }   /* NFL red for negative yards */
        .result-zero { color: var(--text-muted); }

        /* view-mode radio (Actual / Predicted / Both) */
        .view-mode {
            display: inline-flex; align-items: center;
            font-size: 13px; color: var(--text);
            background: #ffffff; border: 1px solid var(--border);
            border-radius: 10px; padding: 7px 12px; white-space: nowrap;
        }
        .view-mode label { cursor: pointer; }
        .view-mode input { accent-color: var(--nfl-blue); }
        ::-webkit-scrollbar { width: 10px; height: 10px; }
        ::-webkit-scrollbar-thumb { background: #c3ccd8; border-radius: 8px; }
        ::-webkit-scrollbar-track { background: transparent; }
    </style>
</head>
<body>
    {%app_entry%}
    <footer>{%config%}{%scripts%}{%renderer%}</footer>
</body>
</html>"""

_initial_game = GAME_IDS[0] if GAME_IDS else None
_initial_play_opts = play_options(_initial_game) if _initial_game else []
_initial_play = _initial_play_opts[0]["value"] if _initial_play_opts else None

# whether the trained predictor + saved models are usable (controls the toggle)
_PRED_READY = predictions_available()
_PRED_NOTE = "" if _PRED_READY else (_PRED["reason"] or "unavailable")


def _game_options() -> list[dict]:
    """Game dropdown options, each prefixed with its train/test/unseen state."""
    marks = {"train": "● train", "test": "● test", "unseen": "○ unseen"}
    opts = []
    for g in GAME_IDS:
        lbl = game_split_label(g)
        opts.append({"label": f"{g}   {marks[lbl]}", "value": g})
    return opts


def _split_badge_children(game_id):
    """Children for the train/test split badge for the current game."""
    lbl = game_split_label(game_id) if game_id is not None else "unseen"
    short, tip, color = SPLIT_META[lbl]
    return [
        html.Span("Data split", className="result-label"),
        html.Span(short, className="result-value",
                  style={"color": color}, title=tip),
    ]


app.layout = html.Div(
    className="app-shell",
    children=[
        html.Div(
            className="app-header",
            children=[
                html.Div("NFL", className="app-shield"),
                html.Div(
                    className="app-header-text",
                    children=[
                        html.H1(
                            className="app-title",
                            children=[
                                "Big Data Bowl ",
                                html.Span("Play Visualiser", className="accent"),
                            ],
                        ),
                        html.P("Next Gen Stats player tracking, frame by frame "
                               "· powered by AWS",
                               className="app-subtitle"),
                    ],
                ),
            ],
        ),
        # controls island
        html.Div(
            className="card controls-card",
            children=[
                html.Div(
                    className="controls-row",
                    children=[
                        html.Div(className="control", children=[
                            html.Span("Game", className="control-label"),
                            dcc.Dropdown(
                                id="game-dropdown",
                                options=_game_options(),
                                value=_initial_game, clearable=False,
                                style={"width": "240px"}),
                        ]),
                        html.Div(className="control", style={"flex": "1 1 300px"},
                                 children=[
                            html.Span("Play", className="control-label"),
                            dcc.Dropdown(
                                id="play-dropdown",
                                options=_initial_play_opts,
                                value=_initial_play, clearable=False),
                        ]),
                        html.Div(className="control", children=[
                            html.Span("View", className="control-label"),
                            dcc.RadioItems(
                                id="view-mode",
                                className="view-mode",
                                options=[
                                    {"label": " Actual", "value": "actual"},
                                    {"label": (" Predicted" if _PRED_READY
                                               else f" Predicted ({_PRED_NOTE})"),
                                     "value": "pred", "disabled": not _PRED_READY},
                                    {"label": " Both", "value": "both",
                                     "disabled": not _PRED_READY},
                                ],
                                value="actual",
                                inputStyle={"marginRight": "5px"},
                                labelStyle={"marginRight": "12px"}),
                        ]),
                        html.Div(id="split-badge", className="result-badge",
                                 style={"marginLeft": "auto"},
                                 children=_split_badge_children(_initial_game)),
                        html.Div(id="result-badges", className="result-badges",
                                 children=_badges_children(
                                     _initial_game, _initial_play)),
                    ],
                ),
            ],
        ),
        # pitch island
        html.Div(
            className="card pitch-card",
            children=[
                dcc.Graph(
                    id="field-graph",
                    config={"displayModeBar": False, "responsive": True},
                    figure=(build_figure(_initial_game, _initial_play)
                            if _initial_play is not None else go.Figure())),
                html.Div(
                    className="pitch-caption",
                    children=[
                        html.Span(className="legend-chip", children=[
                            html.Span(className="dot",
                                      style={"background": TEAM_COLORS[0]}),
                            "Team A"]),
                        html.Span(className="legend-chip", children=[
                            html.Span(className="dot",
                                      style={"background": TEAM_COLORS[1]}),
                            "Team B"]),
                        html.Span(className="legend-chip", children=[
                            html.Span(className="diamond",
                                      style={"background": BALL_COLOR}),
                            "Ball"]),
                        html.Span(className="legend-chip", children=[
                            html.Span(className="dot",
                                      style={"background": "transparent",
                                             "border": f"2px solid {TEXT_MUTED}"}),
                            "Predicted (ghost, in Both view)"]),
                        html.Span("Press ▶ Play to animate · use View to switch "
                                  "between actual tracking and the model's "
                                  "predicted play"),
                        html.Span("Data split: ● train (model saw it) · "
                                  "● test (held out, unseen) · ○ unseen "
                                  "(not in the model's run)",
                                  style={"color": TEXT_MUTED}),
                    ],
                ),
            ],
        ),
        # footer attribution
        html.Div(
            className="app-footer",
            children=[
                html.Span(children=[
                    html.Strong("NFL Big Data Bowl"),
                    " · player tracking visualisation",
                ]),
                html.Span("Powered by AWS · Next Gen Stats"),
            ],
        ),
    ],
)


@app.callback(
    Output("play-dropdown", "options"),
    Output("play-dropdown", "value"),
    Input("game-dropdown", "value"),
)
def _update_plays(game_id):
    if not game_id:
        return [], None
    opts = play_options(game_id)
    value = opts[0]["value"] if opts else None
    return opts, value


@app.callback(
    Output("field-graph", "figure"),
    Output("result-badges", "children"),
    Output("split-badge", "children"),
    Input("game-dropdown", "value"),
    Input("play-dropdown", "value"),
    Input("view-mode", "value"),
)
def _update_figure(game_id, play_id, view_mode):
    if game_id is None or play_id is None:
        return (go.Figure(), _badges_children(None, None),
                _split_badge_children(None))
    mode = view_mode if view_mode in ("actual", "pred", "both") else "actual"
    return (build_figure(game_id, int(play_id), mode=mode),
            _badges_children(game_id, play_id, view_mode=mode),
            _split_badge_children(game_id))


if __name__ == "__main__":
    if not GAME_IDS:
        sys.exit(f"No tracking files found in {TRACKING_DIR}")
    print(f"Loaded {len(GAME_IDS)} games. Starting server...")
    app.run(debug=True, port=8050)
