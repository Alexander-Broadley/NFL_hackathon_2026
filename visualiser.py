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
# Team markers use the two NFL brand accents (red + blue) off the Shield palette.
TEAM_COLORS = ["#d50a0a", "#4b92db"]  # team A (NFL red), team B (NFL blue)
BALL_COLOR = "#a0522d"  # football leather brown

# ---- NFL "Shield" brand palette --------------------------------------------
# Grounded in the official NFL identity: deep navy, NFL red and white, on a
# dark surface so the Big Data Bowl tracking data stays the focus.
NFL_NAVY = "#013369"       # primary NFL navy (the Shield blue)
NFL_RED = "#d50a0a"        # primary NFL red
NFL_BLUE = "#4b92db"       # secondary NFL blue accent

BG_PAGE = "#02172f"        # page background: deep navy black
BG_SURFACE = "#06244a"     # card / island surface: NFL navy
BG_SURFACE_2 = "#0a3161"   # inputs, raised elements
BORDER = "#1b4a86"         # hairline borders (navy tint)
TEXT = "#ffffff"           # primary text (NFL white)
TEXT_MUTED = "#9db6d8"     # secondary text (soft navy-tinted)
ACCENT = "#d50a0a"         # accent: NFL red

FIELD_GREEN = "#123a24"    # turf green on the dark navy field
FIELD_ENDZONE = "#0d2d1b"  # slightly darker end zones
FIELD_LINE = "#4a7a5c"     # subtle field lines on turf


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
                "playResult", "prePenaltyPlayResult", "passResult"]
        try:
            return pd.read_csv(PLAYS_CSV, usecols=cols)
        except ValueError:
            return pd.read_csv(PLAYS_CSV)
    return pd.DataFrame()


PLAYS = load_plays()
GAME_IDS = list_game_ids()


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
    """
    info = {"playResult": None, "prePenaltyPlayResult": None,
            "passResult": None, "playDescription": None}
    if PLAYS.empty:
        return info
    match = PLAYS[(PLAYS["gameId"] == int(game_id)) & (PLAYS["playId"] == int(play_id))]
    if match.empty:
        return info
    row = match.iloc[0]
    for col in ("playResult", "prePenaltyPlayResult"):
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


def _badges_children(game_id, play_id):
    """Children for the outcome-badges row: pass result + two yardage figures."""
    if game_id is None or play_id is None:
        info = {"playResult": None, "prePenaltyPlayResult": None, "passResult": None}
    else:
        info = play_info(game_id, play_id)

    pass_val, pass_cls = _passresult_badge_parts(info["passResult"])
    pre_val, pre_cls = _yardage_badge_parts(info["prePenaltyPlayResult"])
    net_val, net_cls = _yardage_badge_parts(info["playResult"])
    return [
        _badge("Pass result", pass_val, pass_cls),
        _badge("Pre-penalty", pre_val, pre_cls),
        _badge("Play result", net_val, net_cls),
    ]


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


def build_figure(game_id: str, play_id: int) -> go.Figure:
    df = load_tracking(game_id)
    play = df[df["playId"] == play_id].copy()
    if play.empty:
        return go.Figure()

    play.sort_values(["frameId", "nflId"], inplace=True)
    frames_ids = sorted(play["frameId"].unique().tolist())

    # play outcome from plays.csv, matched on gameId+playId
    info = play_info(game_id, play_id)
    result_txt = _result_text(info["playResult"])
    if info["passResult"] is not None:
        pass_label = PASS_RESULT_MAP.get(info["passResult"],
                                         (info["passResult"], ""))[0]
        result_txt = f"{pass_label}  ·  {result_txt}"

    # team ordering: football last; two real teams get stable colours
    teams = [t for t in play["team"].unique().tolist() if t != BALL_LABEL]
    teams = sorted(teams)
    color_for = {t: TEAM_COLORS[i % len(TEAM_COLORS)] for i, t in enumerate(teams)}
    color_for[BALL_LABEL] = BALL_COLOR

    def frame_traces(fid: int) -> list[go.Scatter]:
        f = play[play["frameId"] == fid]
        traces = []
        # one marker trace per team (so the legend shows team names)
        for t in teams:
            sub = f[f["team"] == t]
            traces.append(go.Scatter(
                x=sub["x"], y=sub["y"], mode="markers+text",
                marker=dict(size=16, color=color_for[t],
                            line=dict(color="white", width=1)),
                text=sub["jerseyNumber"].astype("Int64").astype(str),
                textposition="middle center",
                textfont=dict(color="white", size=9),
                name=t, legendgroup=t,
                hovertemplate=(f"{t}<br>#%{{text}}<br>"
                               "x=%{x:.1f}, y=%{y:.1f}<extra></extra>"),
            ))
            # orientation ticks for this team
            if not sub.empty:
                ox, oy = _orientation_segments(sub)
                traces.append(go.Scatter(
                    x=ox, y=oy, mode="lines",
                    line=dict(color=color_for[t], width=1),
                    hoverinfo="skip", showlegend=False, legendgroup=t,
                ))
        # the ball
        ball = f[f["team"] == BALL_LABEL]
        traces.append(go.Scatter(
            x=ball["x"], y=ball["y"], mode="markers",
            marker=dict(size=10, color=BALL_COLOR, symbol="diamond",
                        line=dict(color="white", width=1)),
            name="ball", hovertemplate="ball<br>x=%{x:.1f}, y=%{y:.1f}<extra></extra>",
        ))
        return traces

    # initial data = first frame
    init_traces = frame_traces(frames_ids[0])

    # animation frames
    go_frames = []
    for fid in frames_ids:
        ev = play[play["frameId"] == fid]["event"].iloc[0]
        ev_txt = "" if ev in (None, "None", "nan") else f"  |  event: {ev}"
        go_frames.append(go.Frame(
            data=frame_traces(fid), name=str(fid),
            layout=go.Layout(title_text=_title(game_id, play_id, fid,
                                               len(frames_ids), ev_txt, result_txt)),
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
app.title = "NFL Big Data Bowl · Play Visualiser"

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
            --nfl-navy: #013369;
            --nfl-red: #d50a0a;
            --nfl-blue: #4b92db;
            --bg-page: #02172f;
            --bg-surface: #06244a;
            --bg-surface-2: #0a3161;
            --border: #1b4a86;
            --text: #ffffff;
            --text-muted: #9db6d8;
            --accent: #d50a0a;
        }
        * { box-sizing: border-box; }
        body {
            margin: 0;
            background:
                radial-gradient(1200px 600px at 15% -10%, #0a3161 0%, rgba(10,49,97,0) 60%),
                radial-gradient(1000px 500px at 110% 10%, #013369 0%, rgba(1,51,105,0) 55%),
                var(--bg-page);
            color: var(--text);
            font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
            -webkit-font-smoothing: antialiased;
        }
        .app-shell { max-width: 1120px; margin: 0 auto; padding: 32px 20px 48px; }
        .app-header {
            display: flex; align-items: center; gap: 16px; margin-bottom: 22px;
            padding-bottom: 18px; border-bottom: 1px solid var(--border);
        }
        /* NFL Shield mark rendered inline, no external asset required */
        .nfl-shield {
            flex: 0 0 auto; width: 40px; height: 52px; border-radius: 6px 6px 10px 10px;
            background: linear-gradient(180deg, var(--nfl-navy) 0%, #012349 100%);
            border: 2px solid #ffffff;
            display: flex; align-items: center; justify-content: center;
            color: #ffffff; font-weight: 800; font-size: 14px; letter-spacing: 0.04em;
            box-shadow: 0 4px 14px rgba(1,51,105,0.6);
            clip-path: polygon(0 0, 100% 0, 100% 68%, 50% 100%, 0 68%);
        }
        .app-headings { display: flex; flex-direction: column; gap: 2px; }
        .app-eyebrow {
            font-size: 11px; font-weight: 700; letter-spacing: 0.14em;
            text-transform: uppercase; color: var(--nfl-red); margin: 0;
        }
        .app-title {
            font-size: 24px; font-weight: 800; letter-spacing: -0.01em; margin: 0;
            text-transform: uppercase;
        }
        .app-subtitle { font-size: 13px; color: var(--text-muted); margin: 0; }
        /* footer attribution */
        .app-footer {
            margin-top: 26px; padding-top: 16px; border-top: 1px solid var(--border);
            display: flex; justify-content: space-between; flex-wrap: wrap; gap: 8px;
            font-size: 11px; color: var(--text-muted); letter-spacing: 0.02em;
        }
        .app-footer strong { color: var(--text); font-weight: 700; }
        .card {
            background: var(--bg-surface);
            border: 1px solid var(--border);
            border-radius: 16px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.45), inset 0 1px 0 rgba(255,255,255,0.03);
        }
        .controls-card { padding: 18px 20px; margin-bottom: 20px; }
        .controls-row { display: flex; gap: 18px; flex-wrap: wrap; align-items: flex-end; }
        .control { display: flex; flex-direction: column; gap: 6px; }
        .control-label {
            font-size: 11px; font-weight: 600; letter-spacing: 0.06em;
            text-transform: uppercase; color: var(--text-muted);
        }
        /* the pitch island */
        .pitch-card { padding: 14px 14px 6px; }
        .pitch-caption {
            display: flex; gap: 18px; flex-wrap: wrap;
            padding: 10px 6px 4px; font-size: 12px; color: var(--text-muted);
        }
        .legend-chip { display: inline-flex; align-items: center; gap: 7px; }
        .dot { width: 11px; height: 11px; border-radius: 50%; display: inline-block;
               border: 1px solid rgba(255,255,255,0.4); }
        .diamond { width: 10px; height: 10px; display: inline-block; transform: rotate(45deg);
                   border: 1px solid rgba(255,255,255,0.4); }

        /* Dark-theme the dash dropdowns (react-select) */
        .Select-control, .is-focused:not(.is-open) > .Select-control {
            background: var(--bg-surface-2) !important;
            border: 1px solid var(--border) !important;
            border-radius: 10px !important;
            color: var(--text) !important;
            box-shadow: none !important;
        }
        .Select-menu-outer {
            background: var(--bg-surface-2) !important;
            border: 1px solid var(--border) !important;
            border-radius: 10px !important;
            color: var(--text) !important;
            overflow: hidden;
        }
        .Select-value-label, .Select-placeholder, .Select input > input,
        .Select-value { color: var(--text) !important; }
        .VirtualizedSelectOption, .Select-option {
            background: var(--bg-surface-2) !important;
            color: var(--text) !important;
        }
        .VirtualizedSelectFocusedOption, .Select-option.is-focused {
            background: #0f3d75 !important;
        }
        .Select-arrow { border-color: var(--text-muted) transparent transparent; }

        /* play-outcome badges */
        .result-badges { display: flex; gap: 10px; flex-wrap: wrap; }
        .result-badge {
            display: inline-flex; flex-direction: column; gap: 2px;
            padding: 8px 16px; border-radius: 10px;
            background: var(--bg-surface-2); border: 1px solid var(--border);
            min-width: 120px;
        }
        .result-label {
            font-size: 10px; font-weight: 600; letter-spacing: 0.06em;
            text-transform: uppercase; color: var(--text-muted);
        }
        .result-value { font-size: 20px; font-weight: 700; line-height: 1.1; }
        .result-gain { color: #35b35a; }   /* green for positive yards */
        .result-loss { color: var(--nfl-red); }   /* NFL red for negative yards */
        .result-zero { color: var(--text-muted); }
        ::-webkit-scrollbar { width: 10px; height: 10px; }
        ::-webkit-scrollbar-thumb { background: #0f3d75; border-radius: 8px; }
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

app.layout = html.Div(
    className="app-shell",
    children=[
        html.Div(
            className="app-header",
            children=[
                html.Div("NFL", className="nfl-shield"),
                html.Div(
                    className="app-headings",
                    children=[
                        html.P("Big Data Bowl", className="app-eyebrow"),
                        html.H1("Play Visualiser", className="app-title"),
                        html.P("Next Gen Stats player tracking — frame by frame",
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
                                options=[{"label": g, "value": g} for g in GAME_IDS],
                                value=_initial_game, clearable=False,
                                style={"width": "240px"}),
                        ]),
                        html.Div(className="control", style={"flex": "1 1 360px"},
                                 children=[
                            html.Span("Play", className="control-label"),
                            dcc.Dropdown(
                                id="play-dropdown",
                                options=_initial_play_opts,
                                value=_initial_play, clearable=False),
                        ]),
                        html.Div(id="result-badges", className="result-badges",
                                 style={"marginLeft": "auto"},
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
                        html.Span("Press ▶ Play to animate · tick = player orientation"),
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
    Input("game-dropdown", "value"),
    Input("play-dropdown", "value"),
)
def _update_figure(game_id, play_id):
    if game_id is None or play_id is None:
        return go.Figure(), _badges_children(None, None)
    return build_figure(game_id, int(play_id)), _badges_children(game_id, play_id)


if __name__ == "__main__":
    if not GAME_IDS:
        sys.exit(f"No tracking files found in {TRACKING_DIR}")
    print(f"Loaded {len(GAME_IDS)} games. Starting server...")
    app.run(debug=True, port=8050)
