import visualiser as v

gid = v.GAME_IDS[0]
pid = int(v.play_options(gid)[0]["value"])
fig = v.build_figure(gid, pid, mode="pred")
fid = int(fig.frames[-1].name)
fr = next(f for f in fig.frames if f.name == str(fid))

print(f"=== mode=pred, frame {fid} ===")
for t in fr.data:
    nm = t.name
    mode = t.mode
    sym = getattr(t.marker, "symbol", None) if hasattr(t, "marker") else None
    n = len(t.x) if t.x is not None else 0
    print(f"  trace name={nm!r} mode={mode} symbol={sym} npts={n}")
arrows = [a for a in (fr.layout.annotations or []) if getattr(a, "showarrow", False)]
print("arrow annotations:", len(arrows))
