#!/usr/bin/env python3
"""Render the boxes3d CSV (from `boxes3d`) to a 3D mp4 of the falling/colliding
box pile. Physics is Y-up; we map physics (x, y, z) -> plot (x, z, y) so vertical
is up. Each box is drawn as an oriented cube from its pose quaternion.

Usage: python3 render_boxes3d.py /tmp/boxes3d.csv /tmp/boxes3d.mp4
"""
import sys
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.animation import FuncAnimation, FFMpegWriter

csv = sys.argv[1] if len(sys.argv) > 1 else "/tmp/boxes3d.csv"
out = sys.argv[2] if len(sys.argv) > 2 else "/tmp/boxes3d.mp4"
HALF = 0.5
DT = 1.0 / 60.0

d = np.genfromtxt(csv, delimiter=",", names=True)
steps = np.unique(d["step"]).astype(int)
boxes = np.unique(d["box"]).astype(int)
nb = len(boxes)
# index[step, box] -> row
pos = {}
for row in d:
    pos[(int(row["step"]), int(row["box"]))] = (
        row["x"], row["y"], row["z"], row["qx"], row["qy"], row["qz"], row["qw"]
    )

# local cube corners and its 6 faces (as corner-index quads)
C = HALF * np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
FACES = [(0, 1, 3, 2), (4, 5, 7, 6), (0, 1, 5, 4), (2, 3, 7, 6), (0, 2, 6, 4), (1, 3, 7, 5)]


def quat_R(x, y, z, w):
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def box_faces(px, py, pz, qx, qy, qz, qw):
    R = quat_R(qx, qy, qz, qw)
    w = (R @ C.T).T + np.array([px, py, pz])      # world corners (phys frame)
    w = w[:, [0, 2, 1]]                            # map (x,y,z) -> plot (x,z,y)
    return [[w[i] for i in f] for f in FACES]


cmap = plt.cm.viridis(np.linspace(0.1, 0.95, nb))

fig = plt.figure(figsize=(6.4, 5.6))
ax = fig.add_subplot(projection="3d")
fig.suptitle("nexus GPU rigid-body physics — 18 boxes dropped onto a floor (3D)", fontsize=11)


def draw(step):
    ax.cla()
    ax.set_xlim(-3, 3); ax.set_ylim(-3, 3); ax.set_zlim(0, 5)
    ax.set_box_aspect((1, 1, 0.85))
    ax.view_init(elev=18, azim=-50 + step * 0.15)  # slow orbit
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    # floor
    g = 3.0
    floor = [[(-g, -g, 0), (g, -g, 0), (g, g, 0), (-g, g, 0)]]
    ax.add_collection3d(Poly3DCollection(floor, facecolor="0.85", edgecolor="0.6", alpha=0.6))
    for b in boxes:
        x, y, z, qx, qy, qz, qw = pos[(step, b)]
        faces = box_faces(x, y, z, qx, qy, qz, qw)
        pc = Poly3DCollection(faces, facecolor=cmap[b], edgecolor="k", linewidths=0.4, alpha=0.95)
        ax.add_collection3d(pc)
    ax.text2D(0.02, 0.95, f"t = {step * DT:4.2f} s", transform=ax.transAxes, family="monospace")


anim = FuncAnimation(fig, draw, frames=steps, interval=1000 * DT, blit=False)
anim.save(out, writer=FFMpegWriter(fps=60, bitrate=3000))
print(f"wrote {out} ({len(steps)} frames, {len(steps) * DT:.1f}s, {nb} boxes)")
