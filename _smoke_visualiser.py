"""Headless smoke test for visualiser.py: builds a figure without starting
the Dash server, and checks the playResult / prePenaltyPlayResult / passResult
joins against plays.csv. Run:  python _smoke_visualiser.py
"""
import pandas as pd
import visualiser as v

print("plays.csv loaded cols:", list(v.PLAYS.columns))
for c in ("playResult", "prePenaltyPlayResult", "passResult"):
    print(f"  {c} present:", c in v.PLAYS.columns)

gid = v.GAME_IDS[0]
opts = v.play_options(gid)
pid = opts[0]["value"]
print(f"\ngame {gid}, play {pid}")
print("play_info:", v.play_info(gid, pid))

badges = v._badges_children(gid, pid)
for b in badges:
    label = b.children[0].children
    val = b.children[1].children
    cls = b.children[1].className
    print(f"  badge: {label!r} -> {val!r}  ({cls})")

fig = v.build_figure(gid, int(pid))
print("figure title:", fig.layout.title.text)
print("animation frames:", len(fig.frames))

# cross-check all three fields against plays.csv for several plays
raw = pd.read_csv(v.PLAYS_CSV,
                  usecols=["gameId", "playId", "playResult",
                           "prePenaltyPlayResult", "passResult"])
ok = True
for pid2 in [o["value"] for o in opts[:5]]:
    r = raw[(raw.gameId == int(gid)) & (raw.playId == pid2)].iloc[0]
    want = {
        "playResult": None if pd.isna(r.playResult) else int(r.playResult),
        "prePenaltyPlayResult": None if pd.isna(r.prePenaltyPlayResult) else int(r.prePenaltyPlayResult),
        "passResult": None if pd.isna(r.passResult) else str(r.passResult),
    }
    got = v.play_info(gid, pid2)
    got = {k: got[k] for k in want}
    match = want == got
    ok = ok and match
    print(f"  play {pid2}: want={want} got={got} match={match}")
print("ALL MATCH:", ok)
print("OK")
