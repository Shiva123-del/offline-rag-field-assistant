"""Draws the project thumbnail (1200x627): an operator in the desert with a broken-down
vehicle, using the offline assistant where there is no internet.

    python3 evaluation/make_thumbnail.py      ->  evaluation/assets/thumbnail.png

Pure matplotlib (no image model): every shape is drawn in code. The screen inset shows
the assistant's real answer for "Why does the engine crank but not start?" (manual p. 287).
"""
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Circle, FancyBboxPatch, Polygon, Wedge  # noqa: E402

plt.rcParams["font.family"] = ["Helvetica Neue", "Arial", "DejaVu Sans"]
W, H = 1200, 627
OUT = Path(__file__).resolve().parent / "assets" / "thumbnail.png"

fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W)
ax.set_ylim(H, 0)
ax.axis("off")


def poly(pts, color, z=2, alpha=1.0, ec="none", lw=0):
    ax.add_patch(Polygon(pts, closed=True, fc=color, ec=ec, lw=lw, alpha=alpha, zorder=z))


def circ(x, y, r, color, z=3, alpha=1.0, ec="none", lw=0):
    ax.add_patch(Circle((x, y), r, fc=color, ec=ec, lw=lw, alpha=alpha, zorder=z))


def rrect(x, y, w, h, r, color, z=3, alpha=1.0, ec="none", lw=0):
    ax.add_patch(FancyBboxPatch((x + r, y + r), w - 2 * r, h - 2 * r,
                                boxstyle=f"round,pad={r},rounding_size={r}",
                                fc=color, ec=ec, lw=lw, alpha=alpha, zorder=z))


def limb(p1, p2, width, color, z=4):
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=color, lw=width * 0.72,
            solid_capstyle="round", zorder=z)


def text(x, y, s, size=12, color="white", weight="normal", z=10, ha="left", va="center", alpha=1.0):
    return ax.text(x, y, s, fontsize=size, color=color, fontweight=weight, zorder=z,
                   ha=ha, va=va, alpha=alpha)


# ------------------------------------------------------------------ sky (dusk) + sun
HORIZON = 432
stops = np.array([[0x16, 0x20, 0x45], [0x4b, 0x3f, 0x7a], [0xc2, 0x5f, 0x6e], [0xf0, 0x92, 0x58], [0xfb, 0xc6, 0x73]]) / 255
pos = [0, 0.38, 0.68, 0.88, 1.0]
t = np.linspace(0, 1, 256)
sky = np.stack([np.interp(t, pos, stops[:, i]) for i in range(3)], axis=1)[:, None, :]
ax.imshow(sky, extent=[0, W, HORIZON, 0], aspect="auto", zorder=0)
rng = np.random.default_rng(7)
for _ in range(60):
    circ(rng.uniform(0, W), rng.uniform(0, 230) ** 1.0, rng.uniform(0.5, 1.4), "white", 1, alpha=rng.uniform(0.25, 0.8))
for r, a in ((150, 0.08), (110, 0.12), (80, 0.18), (58, 0.9)):
    circ(905, 412, r, "#ffd98a", 1, alpha=a)

# ------------------------------------------------------------------ dunes and ground
x = np.linspace(0, W, 600)
far = 418 + 14 * np.sin(x / 130 + 1) + 7 * np.sin(x / 47)
mid = 442 + 16 * np.sin(x / 170 + 3) + 6 * np.sin(x / 61 + 1)
ax.fill_between(x, far, H, color="#8b5663", zorder=1)
ax.fill_between(x, mid, H, color="#c27a50", zorder=1)
ground = np.linspace(0, 1, 256)[:, None]
g = (np.array([0xe3, 0xa6, 0x66]) * (1 - ground) + np.array([0xb9, 0x7b, 0x48]) * ground) / 255
ax.imshow(g[:, None, :].repeat(2, axis=1), extent=[0, W, H, 470], aspect="auto", zorder=1)
ax.fill_between(x, 470 + 5 * np.sin(x / 90), 478, color="#e3a666", zorder=1)
for _ in range(120):
    px, py = rng.uniform(0, W), rng.uniform(486, H)
    circ(px, py, rng.uniform(0.8, 2.2), "#f1c48c", 1, alpha=0.5)

# ------------------------------------------------------------------ vehicle (broken down, hood up)
GY = 522
ax.add_patch(plt.matplotlib.patches.Ellipse((285, GY + 2), 470, 24, fc="#5a3a26", alpha=0.45, zorder=2))
# cab and body
poly([(228, 425), (262, 356), (425, 356), (478, 425)], "#566837", 4)
poly([(272, 366), (338, 366), (338, 414), (252, 414)], "#9bb4c2", 5)
poly([(346, 366), (416, 366), (448, 414), (346, 414)], "#9bb4c2", 5)
poly([(272, 366), (300, 366), (262, 414), (252, 414)], "white", 5, alpha=0.18)
ax.plot([262, 428], [350, 350], color="#2a2a2a", lw=1.8, zorder=5)
for rx in (290, 330, 370, 410):
    ax.plot([rx, rx], [350, 356], color="#2a2a2a", lw=1.5, zorder=5)
rrect(90, 424, 402, 70, 8, "#667b40", 4)
rrect(90, 466, 402, 28, 6, "#4a5a2d", 5)
ax.plot([338, 338], [426, 492], color="#3e4c26", lw=1.5, zorder=6)
rrect(318, 440, 14, 5, 2, "#2c2c2c", 6)
rrect(358, 440, 14, 5, 2, "#2c2c2c", 6)
# engine bay + open hood
rrect(98, 424, 134, 24, 3, "#1d1f22", 5)
rrect(112, 428, 42, 17, 3, "#5b6068", 6)
rrect(160, 430, 54, 12, 3, "#7a7f88", 6)
for ex in (118, 130, 142):
    ax.plot([ex, ex], [429, 444], color="#35383d", lw=1.2, zorder=7)
poly([(228, 424), (206, 337), (118, 321), (140, 410)], "#718848", 4)  # raised hood
poly([(228, 424), (206, 337), (214, 338), (236, 424)], "#3e4c26", 4)
ax.plot([140, 224], [405, 417], color="#2a2a2a", lw=2, zorder=5)               # hood prop
# front details
circ(100, 456, 9, "#ffe08a", 6)
circ(100, 456, 9, "none", 6, ec="#3a3a3a", lw=1.2)
rrect(86, 476, 22, 10, 3, "#2c2c2c", 6)
for gy_ in (462, 468, 474):
    ax.plot([112, 130], [gy_, gy_], color="#2a2a2a", lw=1.4, zorder=6)
# antenna, spare wheel
ax.plot([452, 470], [358, 288], color="#222", lw=1.6, zorder=5)
poly([(470, 288), (492, 296), (470, 304)], "#c0392b", 5)
circ(487, 446, 24, "#1c1c1c", 4)
circ(487, 446, 13, "#4a4a4a", 4)
# wheels
for wx in (160, 424):
    ax.add_patch(Wedge((wx, 468), 52, 180, 360, fc="#2f3b1d", zorder=5))
    circ(wx, 500, 43, "#161616", 7)
    circ(wx, 500, 30, "#3a3a3a", 7)
    circ(wx, 500, 13, "#9298a1", 8)
    for k in range(5):
        a = 2 * np.pi * k / 5
        circ(wx + 21 * np.cos(a), 500 + 21 * np.sin(a), 2.3, "#8a8f98", 8)
# smoke
for sx, sy, sr, sa in ((166, 402, 12, 0.7), (150, 372, 17, 0.62), (130, 336, 22, 0.55),
                       (140, 292, 28, 0.45), (112, 250, 34, 0.35), (130, 205, 38, 0.24)):
    circ(sx, sy, sr, "#d9dce1", 9, alpha=sa)
# warning triangle
poly([(48, 548), (68, 548), (58, 528)], "#d94a38", 6)
poly([(53, 545), (63, 545), (58, 534)], "#fff1c9", 6)

# ------------------------------------------------------------------ operator (drawn at 1.3x)
K, AX_, AY_ = 1.3, 640, 522


def P(px, py):
    return (AX_ + (px - AX_) * K, AY_ + (py - AY_) * K)


def prect(x, y, w, h, r, color, z):
    x2, y2 = P(x, y)
    rrect(x2, y2, w * K, h * K, r * K, color, z)


def plimb(p1, p2, width, color, z):
    limb(P(*p1), P(*p2), width * K, color, z)


prect(600, 472, 82, 50, 4, "#a97f4e", 4)                                  # crate
for yy in (490, 506):
    a1, a2 = P(604, yy), P(678, yy)
    ax.plot([a1[0], a2[0]], [a1[1], a2[1]], color="#7e5a33", lw=1.5, zorder=5)
prect(640, 410, 30, 52, 8, "#3d4a2c", 4)                                  # backpack
plimb((645, 468), (595, 468), 24, "#4a5233", 5)                           # thigh
plimb((595, 468), (590, 512), 20, "#4a5233", 5)                           # shin
prect(570, 508, 32, 14, 5, "#262626", 6)                                  # boot
plimb((646, 466), (630, 410), 36, "#7d7a4e", 6)                           # torso
plimb((630, 412), (606, 446), 14, "#7d7a4e", 8)                           # upper arm
plimb((606, 446), (584, 452), 12, "#c68e68", 8)                           # forearm
hx, hy = P(628, 388)
circ(hx, hy, 18 * K, "#c68e68", 7)                                        # head
ax.add_patch(Wedge((hx, hy - 4 * K), 20 * K, 180, 360, fc="#5e5d3a", zorder=8))   # cap
bx_, by_ = P(604, 380)
rrect(bx_, by_, 24 * K, 5 * K, 2, "#4c4b2f", 8)
# laptop
prect(551, 460, 64, 7, 2, "#2b2f36", 9)
poly([P(555, 460), P(562, 460), P(549, 398), P(541, 399)], "#8d949e", 9)
gx, gy = P(585, 432)
for i in range(24):
    circ(gx, gy, 95 * (1 - i / 24), "#ffe3a8", 8, alpha=0.012)

# ------------------------------------------------------------------ screen inset (real UI)
IX, IY, IW, IH = 780, 172, 392, 292
from matplotlib.path import Path as MPath  # noqa: E402
from matplotlib.patches import PathPatch  # noqa: E402
lx, ly = P(549, 400)
ax.add_patch(PathPatch(MPath([(lx, ly), (lx - 10, 300), (700, 250), (IX - 6, 300)],
                             [MPath.MOVETO, MPath.CURVE4, MPath.CURVE4, MPath.CURVE4]),
                       fc="none", ec="#fff3d6", lw=1.6, ls=(0, (4, 3)), zorder=9, alpha=0.9))
circ(lx, ly, 4, "#fff3d6", 9)
rrect(IX - 5, IY - 5, IW + 10, IH + 10, 14, "#10151c", 10)
rrect(IX, IY, IW, IH, 10, "#0d1117", 11)
rrect(IX, IY, IW, 42, 10, "#161b22", 11)
rrect(IX + 12, IY + 8, 26, 26, 6, "#f5a623", 12)
text(IX + 25, IY + 21, "F", 13, "#1a1a1a", "bold", 13, ha="center")
text(IX + 48, IY + 15, "Field Assistant", 11.5, "white", "bold", 13)
text(IX + 48, IY + 31, "HMMWV Operator's Manual", 7.5, "#8b97a8", "normal", 13)
rrect(IX + IW - 82, IY + 12, 70, 20, 10, "#0f2a1c", 12, ec="#2e9e5b", lw=1)
text(IX + IW - 47, IY + 22, "Offline", 8.5, "#4fd283", "normal", 13, ha="center")
rrect(IX + 14, IY + 54, IW - 28, 28, 7, "#161b22", 12)
text(IX + 26, IY + 68, "Why does the engine crank but not start?", 9.5, "white", "bold", 13)
text(IX + 16, IY + 98, "TROUBLESHOOTING TABLE", 7.8, "#f5a623", "bold", 13)
cards = [("Check to see if fuel gauge indicates “E” (empty).", "Fill fuel tank, and start engine."),
         ("Check WAIT-TO-START lamp fails to light.", "Notify unit maintenance."),
         ("Purge fuel system of air (para. 3-10).", "Follow the step as written.")]
cy = IY + 108
for cause, action in cards:
    rrect(IX + 14, cy, IW - 28, 50, 6, "#161b22", 12)
    ax.add_patch(plt.Rectangle((IX + 14, cy + 4), 3, 42, fc="#f5a623", zorder=13))
    text(IX + 26, cy + 11, "3. ENGINE CRANKS BUT DOES NOT START", 6.6, "#8b97a8", "bold", 13)
    text(IX + 26, cy + 26, "CAUSE", 6.6, "#8b97a8", "bold", 13)
    text(IX + 62, cy + 26, cause, 8, "white", "normal", 13)
    text(IX + 26, cy + 40, "ACTION", 6.6, "#8b97a8", "bold", 13)
    text(IX + 66, cy + 40, action, 8, "white", "normal", 13)
    rrect(IX + IW - 82, cy + 33, 58, 14, 7, "#12294a", 13)
    text(IX + IW - 53, cy + 40, "Page 287", 6.8, "#7fb2f0", "bold", 14, ha="center")
    cy += 56
    if cy > IY + IH - 60:
        break

# ------------------------------------------------------------------ title block
text(40, 54, "OFFLINE RAG FIELD ASSISTANT", 31, "white", "bold", 15)
text(41, 96, "Find the fix in the manual.", 19, "#ffd98a", "bold", 15)
text(41, 124, "Even with no internet.", 19, "#ffd98a", "bold", 15)

# no-internet badge (wifi icon with a slash)
bx, by = 1000, 56
rrect(bx - 16, by - 34, 200, 68, 12, "#2a0f12", 15, alpha=0.85, ec="#e5534b", lw=1.6)
for r in (8, 16, 24):
    ax.add_patch(Wedge((bx + 18, by + 12), r, 225, 315, width=3.2, fc="#ff8a80", zorder=16))
circ(bx + 18, by + 14, 3, "#ff8a80", 16)
ax.plot([bx - 2, bx + 40], [by + 22, by - 20], color="#e5534b", lw=3.4, zorder=17, solid_capstyle="round")
text(bx + 54, by - 10, "NO INTERNET", 12.5, "#ff9a92", "bold", 16)
text(bx + 56, by + 12, "no signal needed", 9.5, "#e9c0bc", "normal", 16)

# ------------------------------------------------------------------ bottom strip
chips = ["Works with zero connectivity", "Answers cite manual page numbers", "Runs fully on your own device"]
cx = 40
for c in chips:
    w = 11.2 * len(c) + 34
    rrect(cx, 548, w, 30, 15, "#0f1521", 15, alpha=0.82, ec="#f5a623", lw=1.2)
    circ(cx + 17, 563, 4.5, "#f5a623", 16)
    text(cx + 30, 563, c, 11, "white", "bold", 16)
    cx += w + 14
rrect(0, 592, W, 35, 0, "#0b0f18", 15, alpha=0.9)
text(W / 2, 610, "Independent learning project. Not affiliated with, endorsed or approved by any military or government "
     "organization. Answers are possible causes only, not a diagnosis.", 8.6, "#c2cad8", "normal", 16, ha="center")

OUT.parent.mkdir(exist_ok=True)
fig.savefig(OUT, dpi=100)
print("wrote", OUT)
