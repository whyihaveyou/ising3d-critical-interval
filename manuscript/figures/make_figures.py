#!/usr/bin/env python3
"""Generate the TikZ code of the geometric figures (prism, shapes, path).

Run from this directory:  python3 make_figures.py
The output files fig_prism.tex, fig_shapes.tex and fig_path.tex are
included verbatim by the manuscript.
"""
from itertools import combinations

# ----------------------------------------------------------------- projection
def proj(t, y, z, sx=2.05, sy=1.0, ox=0.62, oy=0.40):
    """Oblique projection: axial coordinate t horizontal, y vertical, z depth."""
    return (sx * t + ox * z, sy * y + oy * z)


def fmt(p):
    return f"({p[0]:.3f},{p[1]:.3f})"


# ----------------------------------------------------------------- the prism
def prism():
    sites = []
    for t in (0, 1):
        for y in (-1, 0, 1):
            for z in (-1, 0, 1):
                sites.append((t, y, z))
    for t in (-1, 2):
        for y, z in ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)):
            sites.append((t, y, z))
    assert len(sites) == 28
    S = set(sites)
    bonds = [(a, b) for a, b in combinations(sites, 2)
             if sum(abs(i - j) for i, j in zip(a, b)) == 1]
    assert len(bonds) == 51
    hubs = {(0, 0, 0), (1, 0, 0)}
    spec = {(0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1), (-1, 0, 0),
            (1, 1, 0), (1, -1, 0), (1, 0, 1), (1, 0, -1), (2, 0, 0)}
    other = S - hubs - spec
    assert len(spec) == 10 and len(other) == 16

    out = [r"\begin{figure}[t]", r"\centering",
           r"\begin{tikzpicture}[scale=1.2,line cap=round]"]
    # back-to-front: sort bonds by mean depth (z larger = further back)
    def depth(b):
        return -(b[0][2] + b[1][2])
    for a, b in sorted(bonds, key=depth):
        if {a, b} == hubs:
            continue
        out.append(rf"\draw[black!45,line width=0.55pt] {fmt(proj(*a))} -- {fmt(proj(*b))};")
    # removed bond
    out.append(rf"\draw[red!70!black,line width=1.4pt,dash pattern=on 3pt off 2pt] "
               rf"{fmt(proj(0,0,0))} -- {fmt(proj(1,0,0))};")
    for s in sorted(S, key=lambda s: -s[2]):
        p = fmt(proj(*s))
        if s in hubs:
            out.append(rf"\fill[black] {p} circle (3.6pt);")
        elif s in spec:
            out.append(rf"\filldraw[fill=white,draw=black,line width=0.9pt] {p} circle (3.2pt);")
        else:
            out.append(rf"\filldraw[fill=black!35,draw=black!60,line width=0.5pt] {p} circle (2.6pt);")
    v, w = proj(0, 0, 0), proj(1, 0, 0)
    out.append(rf"\node[below left=1pt] at {fmt(v)} {{$v$}};")
    out.append(rf"\node[below right=1pt] at {fmt(w)} {{$w$}};")
    out.append(rf"\node[red!70!black,above] at ({(v[0]+w[0])/2:.3f},{(v[1]+w[1])/2:.3f}) {{$e$}};")
    # axis arrows
    o = (-3.6, -2.3)
    out.append(rf"\draw[-{{Stealth[length=4pt]}},thick] {fmt(o)} -- ++(0.9,0) node[right] {{$\delta$}};")
    out.append(rf"\draw[-{{Stealth[length=4pt]}},thick] {fmt(o)} -- ++(0,0.75) node[above] {{$\epsilon_1$}};")
    out.append(rf"\draw[-{{Stealth[length=4pt]}},thick] {fmt(o)} -- ++(0.50,0.33) node[above right=-2pt] {{$\epsilon_2$}};")
    # layer labels
    for t, lab in ((-1, "-1"), (0, "0"), (1, "1"), (2, "2")):
        x, _ = proj(t, -1, 0)
        out.append(rf"\node[font=\footnotesize,black!70] at ({x:.3f},-1.75) {{$t={lab}$}};")
    # legend
    lx, ly = 5.6, 1.1
    out.append(rf"\fill[black] ({lx},{ly}) circle (3.6pt) node[right=5pt,font=\small] {{hubs}};")
    out.append(rf"\filldraw[fill=white,draw=black,line width=0.9pt] ({lx},{ly-0.55}) circle (3.2pt) node[right=5pt,font=\small] {{specified (label $h$)}};")
    out.append(rf"\filldraw[fill=black!35,draw=black!60] ({lx},{ly-1.1}) circle (2.6pt) node[right=5pt,font=\small] {{mark $1$}};")
    out.append(rf"\draw[red!70!black,line width=1.4pt,dash pattern=on 3pt off 2pt] ({lx-0.2},{ly-1.65}) -- ({lx+0.2},{ly-1.65}) node[right=3pt,font=\small,black] {{removed bond}};")
    out += [r"\end{tikzpicture}",
            r"\caption{The prism $\Pi(e)$ of \cref{def:prism}: two $3\times3$ squares at axial "
            r"coordinates $t=0,1$ and two crosses at $t=-1,2$, with all $51$ lattice bonds among the "
            r"$28$ sites except $e$. The label $h(e)$ records which of the ten specified sites belong "
            r"to the path. The remaining $16$ sites are treated as occupied. Bonds between two sites of "
            r"mark $1$ carry the interpolation weight $s$, all other bonds the reference weight $u_0$.}",
            r"\label{fig:prism}", r"\end{figure}"]
    return "\n".join(out) + "\n"


# ----------------------------------------------------------------- shapes
SHAPES = [
    ("S", "straight", [(-1, 0, 0), (0, 0, 0), (1, 0, 0), (2, 0, 0)], "1127558275"),
    ("A", "one axial", [(-1, 0, 0), (0, 0, 0), (1, 0, 0), (1, 1, 0)], "544046375"),
    ("U", "equal", [(0, 1, 0), (0, 0, 0), (1, 0, 0), (1, 1, 0)], "544046375"),
    ("T", "perpendicular", [(0, 1, 0), (0, 0, 0), (1, 0, 0), (1, 0, 1)], "94806903"),
    ("Z", "opposite", [(0, 1, 0), (0, 0, 0), (1, 0, 0), (1, -1, 0)], "0"),
]


def shapes():
    out = [r"\begin{figure}[t]", r"\centering",
           r"\begin{tikzpicture}[scale=0.95,line cap=round,line join=round]"]
    for k, (name, desc, pts, wt) in enumerate(SHAPES):
        dx = 3.3 * k
        P = [proj(*p, sx=0.85, sy=0.85, ox=0.42, oy=0.32) for p in pts]
        P = [(x + dx, y) for x, y in P]
        # faint reference cube edges for depth in the twisted shape
        if name == "T":
            a = proj(1, 0, 0, sx=0.85, sy=0.85, ox=0.42, oy=0.32)
            b = proj(1, 0, 1, sx=0.85, sy=0.85, ox=0.42, oy=0.32)
            c = proj(0, 0, 1, sx=0.85, sy=0.85, ox=0.42, oy=0.32)
            o = proj(0, 0, 0, sx=0.85, sy=0.85, ox=0.42, oy=0.32)
            out.append(rf"\draw[black!20,dashed] ({o[0]+dx:.3f},{o[1]:.3f}) -- ({c[0]+dx:.3f},{c[1]:.3f}) -- ({b[0]+dx:.3f},{b[1]:.3f});")
        if name == "U":
            out.append(rf"\draw[black!50,dash pattern=on 2pt off 1.5pt] {fmt(P[0])} -- {fmt(P[3])};")
        out.append(rf"\draw[black,line width=0.9pt] {fmt(P[0])} -- {fmt(P[1])};")
        out.append(rf"\draw[black,line width=0.9pt] {fmt(P[2])} -- {fmt(P[3])};")
        out.append(rf"\draw[blue!60!black,line width=2.2pt] {fmt(P[1])} -- {fmt(P[2])};")
        for i, p in enumerate(P):
            fill = "black" if i in (1, 2) else "white"
            out.append(rf"\filldraw[fill={fill},draw=black,line width=0.7pt] {fmt(p)} circle (2.4pt);")
        cx = dx + 0.42
        out.append(rf"\node[font=\small] at ({cx:.3f},-1.25) {{$\mathrm{{{name}}}$ ({desc})}};")
        out.append(rf"\node[font=\footnotesize] at ({cx:.3f},-1.7) {{${wt}$}};")
    out += [r"\end{tikzpicture}",
            r"\caption{The five shapes of a triple of consecutive path bonds (\cref{def:shape}). "
            r"The middle bond, whose endpoints are the two hubs, is drawn thick. The numbers are the "
            r"shape charges $10^{12}w_\tau$ of \cref{tab:shapes}. In shape $\mathrm U$ the two outer "
            r"vertices span a chord (dashed).}",
            r"\label{fig:shapes}", r"\end{figure}"]
    return "\n".join(out) + "\n"


# ----------------------------------------------------------------- path
PATH = [(0, 0), (1, 0), (2, 0), (2, 1), (3, 1), (4, 1), (4, 0), (5, 0), (6, 0),
        (6, 1), (6, 2), (5, 2), (4, 2), (3, 2), (2, 2), (2, 3), (3, 3), (4, 3),
        (5, 3), (6, 3), (7, 3), (8, 3), (8, 2), (9, 2)]


def path():
    V = PATH
    n = len(V) - 1
    idx = {p: i for i, p in enumerate(V)}
    adj = lambda a, b: abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1
    chords = [(a, b) for a, b in combinations(V, 2)
              if adj(a, b) and abs(idx[a] - idx[b]) > 1]
    ends = {V[0], V[-1]}
    unch = set(ends)
    for e in ends:
        for p in V:
            if adj(p, e):
                unch.add(p)
    s = 0.78
    P = lambda p: (s * p[0], s * p[1])
    out = [r"\begin{figure}[t]", r"\centering",
           r"\begin{tikzpicture}[line cap=round]"]
    # background grid
    out.append(rf"\draw[black!10,step={s}] (-0.5,-0.5) grid ({s*9+0.5:.2f},{s*3+0.5:.2f});")
    # endpoint neighbourhoods
    for e in ends:
        x, y = P(e); d = s * 1.38
        out.append(rf"\fill[orange!18,rounded corners=3pt] ({x-d:.3f},{y:.3f}) -- ({x:.3f},{y+d:.3f}) -- ({x+d:.3f},{y:.3f}) -- ({x:.3f},{y-d:.3f}) -- cycle;")
    for a, b in chords:
        out.append(rf"\draw[red!70!black,line width=1.0pt,dash pattern=on 2.5pt off 1.8pt] {fmt(P(a))} -- {fmt(P(b))};")
    for i in range(n):
        out.append(rf"\draw[black,line width=1.3pt] {fmt(P(V[i]))} -- {fmt(P(V[i+1]))};")
    for p in V:
        if p in unch:
            out.append(rf"\filldraw[fill=white,draw=black,line width=0.8pt] {fmt(P(p))} circle (2.6pt);")
        else:
            out.append(rf"\fill[black] {fmt(P(p))} circle (2.6pt);")
    out.append(rf"\node[below left=1pt] at {fmt(P(V[0]))} {{$v_0$}};")
    out.append(rf"\node[below right=1pt] at {fmt(P(V[-1]))} {{$v_n$}};")
    out += [r"\end{tikzpicture}",
            rf"\caption{{A path (solid) with its {len(chords)} chords (dashed), drawn in a plane for "
            r"illustration; on $\T_L$ chords may also point out of the plane or across the periodic "
            r"identification. Open circles are the uncharged vertices: the endpoints and their "
            r"lattice neighbours on the path (shaded). Every induced bond at a charged vertex (black) has two "
            r"interior endpoints, so the local rows (A1)--(A2) apply to it.}",
            r"\label{fig:path}", r"\end{figure}"]
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    for name, fn in (("fig_prism.tex", prism), ("fig_shapes.tex", shapes), ("fig_path.tex", path)):
        with open(name, "w") as fh:
            fh.write(fn())
        print("wrote", name)


# ----------------------------------------------------------------- shells
def classify(p):
    a = sorted(abs(c) for c in p)
    n = sum(a)
    if n <= 1:
        return "I"
    if n == 2:
        return "ax" if a == [0, 0, 2] else "face"
    if n == 3:
        if a == [1, 1, 1]:
            return "corner"
        if a == [0, 1, 2]:
            return "link"
        return "ax3"
    return None


STYLE = {
    "I": r"\fill[black] {p} circle (3.4pt);",
    "ax": r"\filldraw[fill=blue!55,draw=blue!40!black] ({x}-0.10,{y}-0.10) rectangle ({x}+0.10,{y}+0.10);",
    "face": r"\filldraw[fill=blue!20,draw=blue!40!black,line width=0.7pt] {p} circle (3.0pt);",
    "corner": r"\filldraw[fill=red!55,draw=red!40!black] ({x},{y}+0.13) -- ({x}-0.12,{y}-0.08) -- ({x}+0.12,{y}-0.08) -- cycle;",
    "link": r"\filldraw[fill=green!50!black!60,draw=green!30!black] ({x},{y}+0.13) -- ({x}+0.13,{y}) -- ({x},{y}-0.13) -- ({x}-0.13,{y}) -- cycle;",
    "ax3": r"\draw[black!60,line width=0.7pt] {p} circle (2.6pt);",
}


def shells():
    s = 0.62
    out = [r"\begin{figure}[t]", r"\centering", r"\begin{tikzpicture}[line cap=round]"]
    for k, z in enumerate((0, 1)):
        ox = 5.2 * k
        pts = [(x, y, z) for x in range(-3, 4) for y in range(-3, 4)
               if classify((x, y, z)) is not None]
        P = {p: (ox + s * p[0], s * p[1]) for p in pts}
        for a, b in combinations(pts, 2):
            if sum(abs(i - j) for i, j in zip(a, b)) == 1:
                hl = {classify(a), classify(b)}
                col = "black!55,line width=0.8pt" if "link" in hl and hl & {"ax", "face"} else "black!18"
                out.append(rf"\draw[{col}] {fmt(P[a])} -- {fmt(P[b])};")
        for p in pts:
            x, y = P[p]
            out.append(STYLE[classify(p)].format(p=fmt((x, y)), x=f"{x:.3f}", y=f"{y:.3f}"))
        out.append(rf"\node[font=\small] at ({ox:.2f},-2.45) {{slice $x_3={z}$}};")
    # legend
    L = [("I", "inner set $I$"), ("ax", "axial shell sites"), ("face", "face shell sites"),
         ("corner", "corners $\\mathcal C$"), ("link", "link sites"), ("ax3", "axial sites $\\pm3e_i$")]
    for i, (c, lab) in enumerate(L):
        x, y = 8.3, 1.55 - 0.55 * i
        out.append(STYLE[c].format(p=fmt((x, y)), x=f"{x:.3f}", y=f"{y:.3f}"))
        out.append(rf"\node[right=4pt,font=\small] at ({x:.2f},{y:.2f}) {{{lab}}};")
    out += [r"\end{tikzpicture}",
            r"\caption{Two horizontal slices of the neighbourhood of the origin used in "
            r"\cref{sec:ordered}. The shell $\partial I=\{\|x\|_1=2\}$ separates the inner set $I$ "
            r"from the exterior. Each link site of type $(2,1,0)$ joins one axial and one face shell "
            r"site (dark bonds); summing over its spin couples the two shell spins and produces the "
            r"factor $\xi^{d(b)}$ in~\eqref{eq:Psi}. The corners appear in the slices $x_3=\pm1$.}",
            r"\label{fig:shells}", r"\end{figure}"]
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    with open("fig_shells.tex", "w") as fh:
        fh.write(shells())
    print("wrote fig_shells.tex")
