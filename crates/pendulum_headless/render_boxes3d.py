#!/usr/bin/env python3
"""Render the boxes3d CSV (from `boxes3d`) to a 3D mp4 of the falling/colliding
box pile. Physics is Y-up; we map physics (x, y, z) -> plot (x, z, y) so vertical
is up. Each box is drawn as an oriented cube from its pose quaternion, with simple
directional shading; all faces share one depth-sorted collection so the cubes
occlude each other correctly.

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
LIGHT = np.array([0.35, 0.9, 0.25])      # light direction in physics frame (+y up)
LIGHT = LIGHT / np.linalg.norm(LIGHT)
AMBIENT, DIFFUSE = 0.45, 0.55

d = np.genfromtxt(csv, delimiter=",", names=True)
steps = np.unique(d["step"]).astype(int)
boxes = np.unique(d["box"]).astype(int)
nb = len(boxes)
pos = {}
for row in d:
    pos[(int(row["step"]), int(row["box"]))] = (
        row["x"], row["y"], row["z"], row["qx"], row["qy"], row["qz"], row["qw"]
    )

# cube corners; 6 faces as corner quads + each face's local outward normal
C = HALF * np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
FACES = [(0, 1, 3, 2), (4, 5, 7, 6), (0, 1, 5, 4), (2, 3, 7, 6), (0, 2, 6, 4), (1, 3, 7, 5)]
NORMALS = np.array([[-1, 0, 0], [1, 0, 0], [0, -1, 0], [0, 1, 0], [0, 0, -1], [0, 0, 1.0]])
PERM = [0, 2, 1]  # physics (x,y,z) -> plot (x,z,y)


def quat_R(x, y, z, w):
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


base = plt.cm.turbo(np.linspace(0.05, 0.95, nb))[:, :3]

fig = plt.figure(figsize=(6.4, 5.6))
fig.patch.set_facecolor("#0e1117")
ax = fig.add_subplot(projection="3d")
ax.set_facecolor("#0e1117")
fig.suptitle("nexus GPU rigid-body physics — 18 boxes onto a floor (3D)",
             color="white", fontsize=11)


def draw(step):
    ax.cla()
    ax.set_xlim(-3, 3); ax.set_ylim(-3, 3); ax.set_zlim(0, 5)
    ax.set_box_aspect((1, 1, 0.85))
    ax.view_init(elev=20, azim=-55 + step * 0.12)  # gentle orbit
    ax.set_axis_off()

    # floor grid
    g = 3.0
    ax.add_collection3d(Poly3DCollection(
        [[(-g, -g, 0), (g, -g, 0), (g, g, 0), (-g, g, 0)]],
        facecolor="#1b2230", edgecolor="none"))
    for t in np.linspace(-g, g, 7):
        ax.plot([-g, g], [t, t], [0, 0], color="#2c3650", lw=0.6)
        ax.plot([t, t], [-g, g], [0, 0], color="#2c3650", lw=0.6)

    verts, facecolors = [], []
    for b in boxes:
        x, y, z, qx, qy, qz, qw = pos[(step, b)]
        R = quat_R(qx, qy, qz, qw)
        w = (R @ C.T).T + np.array([x, y, z])     # world corners (physics frame)
        wn = (R @ NORMALS.T).T                     # world normals (physics frame)
        wp = w[:, PERM]                            # -> plot frame for drawing
        for fi, f in enumerate(FACES):
            shade = AMBIENT + DIFFUSE * max(0.0, float(wn[fi] @ LIGHT))
            verts.append([wp[i] for i in f])
            facecolors.append(np.clip(base[b] * shade, 0, 1))
    pc = Poly3DCollection(verts, facecolors=facecolors,
                          edgecolor=(0, 0, 0, 0.35), linewidths=0.3)
    ax.add_collection3d(pc)
    ax.text2D(0.03, 0.95, f"t = {step * DT:4.2f} s", transform=ax.transAxes,
              family="monospace", color="white")


anim = FuncAnimation(fig, draw, frames=steps, interval=1000 * DT, blit=False)
anim.save(out, writer=FFMpegWriter(fps=60, bitrate=3000),
          savefig_kwargs={"facecolor": "#0e1117"})
print(f"wrote {out} ({len(steps)} frames, {len(steps) * DT:.1f}s, {nb} boxes)")
