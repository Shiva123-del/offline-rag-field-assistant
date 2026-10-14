"""Draws the project thumbnail (1200x627, also saved at 2x): an operator in the desert with a
broken-down vehicle, using the offline assistant where there is no internet.

    python3 evaluation/make_thumbnail.py   ->  evaluation/assets/thumbnail.png (+ thumbnail_hd.png)

Pure matplotlib (no image model): every shape is drawn in code. The screen inset shows the
assistant's real answer for "Why does the engine crank but not start?" (manual p. 287).
"""
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.patheffects as pe  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Circle, Ellipse, FancyBboxPatch, Polygon, Wedge  # noqa: E402

plt.rcParams["font.family"] = ["Helvetica Neue", "Arial", "DejaVu Sans"]
W, H = 1200, 627
ASSETS = Path(__file__).resolve().parent / "assets"
rng = np.random.default_rng(11)

fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W)
ax.set_ylim(H, 0)
ax.axis("off")


def hexrgb(c):
    c = c.lstrip("#")
    return np.array([int(c[i:i + 2], 16) for i in (0, 2, 4)]) / 255


def poly(pts, color, z=2, alpha=1.0, ec="none", lw=0):
    ax.add_patch(Polygon(pts, closed=True, fc=color, ec=ec, lw=lw, alpha=alpha, zorder=z))


def circ(x, y, r, color, z=3, alpha=1.0, ec="none", lw=0):
    ax.add_patch(Circle((x, y), r, fc=color, ec=ec, lw=lw, alpha=alpha, zorder=z))


def rrect(x, y, w, h, r, color, z=3, alpha=1.0, ec="none", lw=0):
    ax.add_patch(FancyBboxPatch((x + r, y + r), w - 2 * r, h - 2 * r,
                                boxstyle=f"round,pad={r},rounding_size={r}",
                                fc=color, ec=ec, lw=lw, alpha=alpha, zorder=z))


def limb(p1, p2, width, color, z=4, alpha=1.0):
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=color, lw=width * 0.72,
            solid_capstyle="round", zorder=z, alpha=alpha)


def text(x, y, s, size=12, color="white", weight="normal", z=10, ha="left", va="center", alpha=1.0):
    return ax.text(x, y, s, fontsize=size, color=color, fontweight=weight, zorder=z,
                   ha=ha, va=va, alpha=alpha)


def shade(pts, c_top, c_bot, z=4, alpha=1.0):
    """Polygon with a vertical colour gradient (gives metal / sky a sense of volume)."""
    patch = Polygon(pts, closed=True, fc="none", ec="none", zorder=z)
    ax.add_patch(patch)
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    t = np.linspace(0, 1, 48)[:, None]
    img = (hexrgb(c_top) * (1 - t) + hexrgb(c_bot) * t)[:, None, :].repeat(2, axis=1)
    im = ax.imshow(img, extent=[min(xs), max(xs), max(ys), min(ys)], aspect="auto", zorder=z, alpha=alpha)
    im.set_clip_path(patch)
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)


def glow(x, y, r, color, z=2, strength=0.5, steps=26):
    for i in range(steps):
        circ(x, y, r * (1 - i / steps), color, z, alpha=strength / steps)


# ------------------------------------------------------------------ sky, sun, ridges
HORIZON = 436
stops = np.array([[0x0e, 0x16, 0x38], [0x35, 0x2f, 0x6b], [0x8a, 0x4a, 0x84], [0xe0, 0x6f, 0x6a],
                  [0xf6, 0xa1, 0x5c], [0xff, 0xd2, 0x80]]) / 255
pos = [0, 0.3, 0.55, 0.76, 0.9, 1.0]
t = np.linspace(0, 1, 256)
sky = np.stack([np.interp(t, pos, stops[:, i]) for i in range(3)], axis=1)[:, None, :]
ax.imshow(sky, extent=[0, W, HORIZON, 0], aspect="auto", zorder=0)
for _ in range(90):
    circ(rng.uniform(0, W), rng.uniform(0, 250), rng.uniform(0.4, 1.3), "white", 1, alpha=rng.uniform(0.2, 0.85))
SUN = (70, 418)
glow(*SUN, 380, "#ff9d57", 1, 0.45)
glow(*SUN, 190, "#ffc27a", 1, 0.6)
circ(*SUN, 62, "#fff0c4", 1, alpha=0.95)
for bx_, by_, bs in ((900, 120, 1.0), (930, 100, 0.8), (960, 128, 0.7)):      # distant birds
    ax.plot([bx_ - 9 * bs, bx_, bx_ + 9 * bs], [by_ - 4 * bs, by_, by_ - 4 * bs], color="#1b1630",
            lw=1.6, zorder=2, solid_capstyle="round", alpha=0.7)
xs_ = np.linspace(0, W, 700)
ridge1 = 408 + 18 * np.sin(xs_ / 90 + 0.5) + 11 * np.sin(xs_ / 33) + 14 * np.sin(xs_ / 210 + 2)
ridge2 = 424 + 12 * np.sin(xs_ / 70 + 2) + 6 * np.sin(xs_ / 25)
ax.fill_between(xs_, ridge1, H, color="#6a4a78", alpha=0.85, zorder=1)
ax.fill_between(xs_, ridge2, H, color="#8c5368", zorder=1)
hz = np.linspace(0, 1, 64)[:, None]
haze = np.dstack([np.ones((64, 1)) * hexrgb("#ffb878")[i] for i in range(3)] + [(1 - hz) * 0.0 + hz * 0.0 + (hz[::-1]) * 0.35])
ax.imshow(haze, extent=[0, W, 452, 390], aspect="auto", zorder=1)

# ------------------------------------------------------------------ ground
dune = 446 + 12 * np.sin(xs_ / 150 + 3) + 5 * np.sin(xs_ / 55 + 1)
ax.fill_between(xs_, dune, H, color="#c47f52", zorder=1)
gr = np.linspace(0, 1, 256)[:, None]
gc = (hexrgb("#e9ad6c") * (1 - gr) + hexrgb("#b57844") * gr)[:, None, :].repeat(2, axis=1)
ax.imshow(gc, extent=[0, W, H, 472], aspect="auto", zorder=1)
ax.fill_between(xs_, 472 + 4 * np.sin(xs_ / 80), 480, color="#e9ad6c", zorder=1)
for k in range(14):                                                           # dune ripples
    yy = 486 + k * 10
    ax.plot(xs_[::6], yy + 2.2 * np.sin(xs_[::6] / (28 + k * 2) + k), color="#a8693b", lw=0.8, alpha=0.22, zorder=1)
for tx in (-60, -30):                                                         # tyre tracks leading in
    ax.plot([1010 + tx, 760 + tx * 0.6, 560, 470], [H + 2, 575, 540, 528], color="#9a6238", lw=3.2, alpha=0.30,
            zorder=1, solid_capstyle="round")
for _ in range(10):                                                           # rocks and tufts
    rx, ry = rng.uniform(20, 1180), rng.uniform(500, 600)
    poly([(rx - 9, ry), (rx - 4, ry - 7), (rx + 5, ry - 6), (rx + 10, ry)], "#8e5b37", 2, alpha=0.8)
for _ in range(9):
    tx, ty = rng.uniform(20, 1180), rng.uniform(496, 606)
    for dx in (-5, -2, 1, 4):
        ax.plot([tx, tx + dx], [ty, ty - rng.uniform(7, 12)], color="#7d5a2c", lw=1.1, zorder=2, alpha=0.8)

# ------------------------------------------------------------------ vehicle (broken down, hood up)
GY = 524
ax.add_patch(Ellipse((330, GY + 5), 560, 22, fc="#4b2e1d", alpha=0.5, zorder=2))        # cast shadow (sun is left)
ax.add_patch(Ellipse((430, GY + 7), 700, 12, fc="#4b2e1d", alpha=0.22, zorder=2))
OL, OD = "#7a8f4c", "#46562a"
shade([(250, 432), (270, 372), (476, 372), (494, 432)], "#6c8044", "#4f6030", 4)         # cab
shade([(280, 381), (352, 381), (352, 422), (264, 422)], "#f4b979", "#7aa0b8", 5)         # side windows reflecting dusk
shade([(360, 381), (462, 381), (476, 422), (360, 422)], "#f4b979", "#7aa0b8", 5)
poly([(284, 381), (312, 381), (276, 422), (264, 422)], "white", 6, alpha=0.2)
ax.plot([268, 484], [366, 366], color="#2a2a2a", lw=2.2, zorder=5)                      # roof rack
for rx in (292, 332, 372, 412, 452):
    ax.plot([rx, rx], [366, 372], color="#2a2a2a", lw=1.5, zorder=5)
rrect(398, 350, 24, 16, 2, "#b3382c", 5)                                                 # jerry can on rack
rrect(404, 345, 8, 5, 1, "#8a2a21", 5)
shade([(78, 454), (86, 430), (252, 428), (505, 432), (516, 456), (516, 492), (78, 492)], OL, OD, 4)   # body
poly([(78, 470), (516, 470), (516, 492), (78, 492)], "#3b4825", 5, alpha=0.7)
ax.plot([354, 354], [434, 490], color="#34401f", lw=1.4, zorder=6)                      # door seam
rrect(328, 444, 15, 5, 2, "#222", 6)
rrect(372, 444, 15, 5, 2, "#222", 6)
rrect(256, 392, 9, 14, 2, "#222", 6)                                                      # mirror
for _ in range(26):                                                                       # mud splatter
    ax.add_patch(Ellipse((rng.uniform(90, 510), rng.uniform(470, 490)), rng.uniform(3, 9), rng.uniform(2, 5),
                         fc="#7a5530", alpha=rng.uniform(0.25, 0.5), zorder=6))
# open hood + engine bay
rrect(90, 428, 150, 26, 3, "#17191c", 5)
rrect(104, 432, 46, 18, 3, "#6a7078", 6)
rrect(158, 434, 58, 12, 3, "#8a9099", 6)
ax.add_patch(plt.Circle((128, 441), 6, fc="#2d3036", ec="#9aa0a8", lw=1.2, zorder=7))
ax.plot([150, 205], [440, 438], color="#c2392b", lw=1.6, zorder=7)
shade([(250, 430), (226, 334), (122, 318), (146, 412)], "#96ad62", "#5b6d37", 4)         # raised hood
poly([(250, 430), (226, 334), (236, 336), (260, 430)], "#2f3b1a", 4)
ax.plot([146, 238], [404, 420], color="#2a2a2a", lw=2.2, zorder=5)                      # hood prop
# front, rear details
circ(92, 458, 10, "#fff1b8", 6)
circ(92, 458, 10, "none", 6, ec="#2f2f2f", lw=1.4)
rrect(76, 480, 26, 11, 3, "#2a2a2a", 6)
for gy_ in (462, 469, 476):
    ax.plot([108, 130], [gy_, gy_], color="#262626", lw=1.5, zorder=6)
glow(512, 450, 22, "#ffb347", 7, 0.9)                                                    # hazard light
circ(512, 450, 4.5, "#ffd27d", 8)
circ(510, 450, 26, "#1b1b1b", 5)                                                         # spare wheel
circ(510, 450, 14, "#444", 5)
ax.plot([458, 478], [372, 296], color="#222", lw=1.6, zorder=5)                          # antenna + flag
poly([(478, 296), (500, 304), (478, 312)], "#c0392b", 5)
# wheels
for wx in (162, 432):
    ax.add_patch(Wedge((wx, 470), 56, 180, 360, fc="#27311a", zorder=5))
    circ(wx, 503, 46, "#131313", 7)
    for k in range(40):
        a = 2 * np.pi * k / 40
        ax.plot([wx + 41 * np.cos(a), wx + 46 * np.cos(a)], [503 + 41 * np.sin(a), 503 + 46 * np.sin(a)],
                color="#2b2b2b", lw=1.6, zorder=8)
    circ(wx, 503, 34, "#262626", 8)
    circ(wx, 503, 24, "#a7adb6", 8)
    circ(wx, 503, 20, "#5d6269", 8)
    for k in range(8):
        a = 2 * np.pi * k / 8
        circ(wx + 14 * np.cos(a), 503 + 14 * np.sin(a), 2.4, "#1d1f22", 9)
    circ(wx, 503, 6, "#2a2d31", 9)
# smoke (lit from the left by the sun)
for sx, sy, sr, sa in ((172, 404, 12, 0.75), (158, 374, 17, 0.68), (140, 338, 22, 0.6), (150, 294, 28, 0.5),
                       (126, 252, 34, 0.4), (148, 206, 40, 0.3)):
    circ(sx, sy, sr, "#e5e1dc", 9, alpha=sa)
    circ(sx + sr * 0.25, sy + sr * 0.2, sr * 0.8, "#b9b3b0", 9, alpha=sa * 0.5)
# tools and warning triangle in front of the vehicle
rrect(26, 520, 52, 22, 3, "#c0392b", 6)
rrect(30, 514, 44, 8, 2, "#8f2b20", 6)
limb((82, 534), (118, 540), 4, "#9aa0a8", 6)
poly([(50, 566), (72, 566), (61, 543)], "#e04a37", 7)
poly([(55, 562), (67, 562), (61, 550)], "#fff1c9", 7)

# ------------------------------------------------------------------ operator (drawn at 1.35x, rim-lit by the sun)
K, AX_, AY_ = 1.35, 650, 524


def P(px, py):
    return (AX_ + (px - AX_) * K + 48, AY_ + (py - AY_) * K)


def plimb(p1, p2, width, color, z, rim=True):
    a, b = P(*p1), P(*p2)
    if rim:
        limb((a[0] - 2.2, a[1] - 0.6), (b[0] - 2.2, b[1] - 0.6), width * K + 1.5, "#ffb06a", z - 0.2, 0.75)
    limb(a, b, width * K, color, z)


def prect(x, y, w, h, r, color, z):
    x2, y2 = P(x, y)
    rrect(x2, y2, w * K, h * K, r * K, color, z)


ax.add_patch(Ellipse((738, AY_ + 4), 330, 13, fc="#4b2e1d", alpha=0.45, zorder=2))    # long shadow to the right
prect(604, 474, 86, 52, 4, "#a97f4e", 4)                                              # crate
for yy in (492, 509):
    a1, a2 = P(608, yy), P(686, yy)
    ax.plot([a1[0], a2[0]], [a1[1], a2[1]], color="#7e5a33", lw=1.5, zorder=5)
prect(646, 412, 34, 56, 9, "#3b4829", 4)                                              # backpack
prect(650, 405, 26, 12, 5, "#2f3a21", 4)                                              # bedroll
plimb((650, 470), (598, 470), 25, "#4d5636", 5)                                       # thigh
plimb((598, 470), (592, 514), 20, "#4d5636", 5)                                       # shin
x1, y1 = P(594, 480)
rrect(x1 - 5, y1 - 3, 14 * K, 12 * K, 4, "#3a4228", 6)                                # knee pad
prect(572, 510, 34, 15, 5, "#262626", 6)                                              # boot
plimb((650, 468), (634, 412), 38, "#8a8654", 6)                                       # torso
plimb((650, 462), (636, 428), 30, "#38432a", 7, rim=False)                            # chest rig
plimb((630, 416), (606, 448), 15, "#8a8654", 8)                                       # upper arm
plimb((606, 448), (586, 454), 12, "#c68e68", 8, rim=False)                            # forearm
plimb((590, 453), (584, 455), 14, "#2e2e2e", 9, rim=False)                            # glove
hx, hy = P(632, 392)
circ(hx, hy, 18.5 * K, "#c68e68", 7)                                                  # head
ax.add_patch(Wedge((hx, hy - 3 * K), 21.5 * K, 175, 365, fc="#4a5233", zorder=8))     # helmet
ax.add_patch(Wedge((hx, hy - 3 * K), 21.5 * K, 175, 215, fc="#ffb06a", alpha=0.7, zorder=8.1))
gx_, gy_ = P(610, 392)
rrect(gx_, gy_, 22 * K, 7 * K, 3, "#16181b", 9)                                       # goggles
rrect(gx_ + 2, gy_ + 1.5, 9 * K, 4 * K, 2, "#7fc4ff", 9.1, alpha=0.8)
ax.plot([hx, hx + 12 * K], [hy + 14 * K, hy + 24 * K], color="#2b2b2b", lw=1.4, zorder=9)  # chin strap
# rugged laptop with cool screen light
prect(552, 462, 66, 8, 2, "#44503a", 10)
a1, a2, a3, a4 = P(556, 462), P(564, 462), P(550, 398), P(541, 399)
poly([a1, a2, a3, a4], "#5b6a45", 10)
gx2, gy2 = P(584, 436)
glow(gx2, gy2, 125, "#b9defa", 8, 0.34, 30)
glow(gx2 + 8, gy2 - 4, 55, "#e8f4ff", 8, 0.35, 20)


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
OUTLINE = [pe.withStroke(linewidth=4, foreground="#0b1020", alpha=0.55)]
text(40, 54, "OFFLINE RAG FIELD ASSISTANT", 31, "white", "bold", 15).set_path_effects(OUTLINE)
text(41, 96, "Find the fix in the manual.", 19, "#ffd98a", "bold", 15).set_path_effects(OUTLINE)
text(41, 124, "Even with no internet.", 19, "#ffd98a", "bold", 15).set_path_effects(OUTLINE)

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

# ------------------------------------------------------------------ vignette + export
yy, xx = np.mgrid[0:64, 0:96]
d = np.sqrt(((xx - 48) / 48) ** 2 + ((yy - 32) / 32) ** 2)
vig = np.dstack([np.zeros((64, 96, 3)), np.clip((d - 0.75) * 0.5, 0, 0.45)])
ax.imshow(vig, extent=[0, W, H, 0], aspect="auto", zorder=20)
ax.set_xlim(0, W)
ax.set_ylim(H, 0)

ASSETS.mkdir(exist_ok=True)
hd = ASSETS / "thumbnail_hd.png"
fig.savefig(hd, dpi=200)
from PIL import Image  # noqa: E402
Image.open(hd).convert("RGB").resize((W, H), Image.LANCZOS).save(ASSETS / "thumbnail.png", optimize=True)
print("wrote", ASSETS / "thumbnail.png", "and thumbnail_hd.png")
