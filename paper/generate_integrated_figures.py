from __future__ import annotations

from pathlib import Path
import csv
import math

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

ROOT = Path(__file__).resolve().parent
FIG_DIR = ROOT / "figures"
PNG_DIR = ROOT / "figures_png"
DATA_DIR = ROOT / "data"
FIG_DIR.mkdir(exist_ok=True)
PNG_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)

INK = "#17212B"
MUTED = "#66727D"
GRID = "#D8E0E6"
NAVY = "#22536F"
BLUE = "#3C7DA7"
TEAL = "#2A8F83"
ORANGE = "#D66747"
GOLD = "#C9952F"
PALE_BLUE = "#EAF2F7"
PALE_TEAL = "#EAF6F3"
PALE_ORANGE = "#FAEEE9"
PALE_GOLD = "#FBF5E7"
PALE_GRAY = "#F2F4F6"
WHITE = "#FFFFFF"

mpl.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 7.1,
        "axes.titlesize": 7.8,
        "axes.labelsize": 6.9,
        "xtick.labelsize": 6.1,
        "ytick.labelsize": 6.1,
        "legend.fontsize": 5.8,
        "axes.linewidth": 0.65,
        "grid.linewidth": 0.45,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.facecolor": WHITE,
    }
)


def save_pair(fig: plt.Figure, stem: str) -> None:
    fig.savefig(FIG_DIR / f"{stem}.pdf", format="pdf", bbox_inches=None, pad_inches=0)
    fig.savefig(PNG_DIR / f"{stem}.png", format="png", dpi=360, bbox_inches=None, pad_inches=0)
    plt.close(fig)


def write_csv(name: str, header: list[str], rows: list[list[object]]) -> None:
    with (DATA_DIR / name).open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def box(ax, xy, wh, title, subtitle=None, *, fc=WHITE, ec=INK,
        title_color=INK, lw=0.9, title_size=6.7, subtitle_size=5.1,
        radius=0.014, title_y=0.64, z=3):
    x, y = xy
    w, h = wh
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.006,rounding_size={radius}",
        fc=fc, ec=ec, lw=lw, zorder=z,
    )
    ax.add_patch(patch)
    if subtitle is None:
        ax.text(x + w/2, y + h/2, title, ha="center", va="center",
                fontsize=title_size, fontweight="bold", color=title_color, zorder=z+1)
    else:
        ax.text(x + w/2, y + h*title_y, title, ha="center", va="center",
                fontsize=title_size, fontweight="bold", color=title_color, zorder=z+1)
        ax.text(x + w/2, y + h*0.28, subtitle, ha="center", va="center",
                fontsize=subtitle_size, color=MUTED, linespacing=1.15, zorder=z+1)
    return patch


def arrow(ax, start, end, *, color=INK, lw=1.25, ls="-", rad=0.0,
          ms=8, z=5, arrowstyle="-|>"):
    patch = FancyArrowPatch(
        start, end, arrowstyle=arrowstyle, mutation_scale=ms,
        color=color, lw=lw, linestyle=ls,
        connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0,
        zorder=z,
    )
    ax.add_patch(patch)
    return patch


# -----------------------------------------------------------------------------
# Figure: decode operating regimes (all analytical, not measured)
# -----------------------------------------------------------------------------
P = 70e9
b_w = 2.0
Ktok = 2 * 80 * 8 * 128 * 2  # bytes/token, GQA BF16
B = np.logspace(0, math.log10(2048), 320)
contexts = [2048, 8192, 32768]
Ibal = 299.0

fig = plt.figure(figsize=(7.15, 2.52))
ax1 = fig.add_axes([0.055, 0.20, 0.275, 0.68])
ax2 = fig.add_axes([0.385, 0.20, 0.275, 0.68])
ax3 = fig.add_axes([0.715, 0.20, 0.255, 0.68])

# (a) bytes per useful output token vs batch
for x in [ax1, ax2, ax3]:
    x.grid(True, which="both", color=GRID, alpha=0.8)
    x.spines[["top", "right"]].set_visible(False)

weight_per_token = P*b_w/B/1e9
kv_per_token_8k = np.full_like(B, 8192*Ktok/1e9)
ax1.loglog(B, weight_per_token, color=NAVY, lw=1.5, label="weights, shared")
ax1.loglog(B, kv_per_token_8k, color=ORANGE, lw=1.5, ls="--", label="KV, private (8K)")
Bc = P*b_w/(8192*Ktok)
ax1.axvline(Bc, color=MUTED, lw=0.8, ls=":")
ax1.scatter([Bc], [P*b_w/Bc/1e9], s=14, facecolor=WHITE, edgecolor=INK, zorder=6)
ax1.text(Bc*1.07, 0.72, f"cross-over\nB≈{Bc:.0f}", fontsize=5.2, color=MUTED)
ax1.set_title("(a) Traffic per output token", loc="left", fontweight="bold")
ax1.set_xlabel("Decode batch size B")
ax1.set_ylabel("Bytes moved (GB/token)")
ax1.set_xlim(1, 2048)
ax1.set_ylim(0.35, 190)
ax1.legend(loc="upper right", frameon=True)

# (b) arithmetic intensity including KV saturation
weights_only = 2*B/b_w
ax2.loglog(B, weights_only, color=INK, lw=1.4, label="weights only")
line_styles = ["--", "-.", ":"]
line_colors = [BLUE, TEAL, ORANGE]
rows_ai: list[list[object]] = []
for S, ls, c in zip(contexts, line_styles, line_colors):
    I = 2*P*B/(P*b_w + B*S*Ktok)
    ax2.loglog(B, I, color=c, lw=1.45, ls=ls, label=f"S={S//1024}K")
    for b_sample in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]:
        i_val = 2*P*b_sample/(P*b_w + b_sample*S*Ktok)
        rows_ai.append([b_sample, S, i_val])
ax2.axhline(Ibal, color=GOLD, lw=1.0)
ax2.text(1.35, Ibal*1.12, "illustrative balance point\n299 FLOP/byte", fontsize=5.1, color=GOLD)
ax2.scatter([64], [2*P*64/(P*b_w+64*8192*Ktok)], s=15, facecolor=WHITE, edgecolor=INK, zorder=7)
ax2.text(75, 20.3, "B=64, S=8K\n28.7 FLOP/byte", fontsize=5.0, color=MUTED)
ax2.set_title("(b) Decode arithmetic intensity", loc="left", fontweight="bold")
ax2.set_xlabel("Decode batch size B")
ax2.set_ylabel("FLOP/byte")
ax2.set_xlim(1, 2048)
ax2.set_ylim(0.8, 2200)
ax2.legend(loc="lower right", frameon=True, ncol=1)

# (c) weight/KV traffic vs context at B=64
S = np.logspace(math.log10(256), math.log10(131072), 300)
weights_step = np.full_like(S, P*b_w/1e9)
kv_step = 64*S*Ktok/1e9
total_step = weights_step + kv_step
Sstar = P*b_w/(64*Ktok)
ax3.loglog(S, weights_step, color=NAVY, lw=1.4, label="weights")
ax3.loglog(S, kv_step, color=ORANGE, lw=1.4, ls="--", label="KV")
ax3.loglog(S, total_step, color=INK, lw=1.0, ls=":", label="total")
ax3.axvline(Sstar, color=MUTED, lw=0.8, ls=":")
ax3.scatter([Sstar], [P*b_w/1e9], s=14, facecolor=WHITE, edgecolor=INK, zorder=6)
ax3.text(Sstar*1.08, 82, f"S*≈{Sstar/1000:.1f}K", fontsize=5.2, color=MUTED)
ax3.set_title("(c) Cache overtakes model", loc="left", fontweight="bold")
ax3.set_xlabel("Context length S")
ax3.set_ylabel("Traffic per decode step (GB)")
ax3.set_xlim(256, 131072)
ax3.set_ylim(4, 5000)
ax3.legend(loc="upper left", frameon=True)

fig.text(0.51, 0.035,
         "Analytical 70B-class example: P=70B, BF16 weights, 80 layers, 8 KV heads, head dim 128, BF16 KV. Not measured.",
         ha="center", va="bottom", fontsize=5.6, color=MUTED)
save_pair(fig, "fig2_decode_regimes")

write_csv(
    "fig2_decode_regimes_ai.csv",
    ["batch_size", "context_tokens", "arithmetic_intensity_FLOP_per_byte"],
    rows_ai,
)
rows_cross = []
for s in [256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]:
    rows_cross.append([s, P*b_w/1e9, 64*s*Ktok/1e9, (P*b_w+64*s*Ktok)/1e9])
write_csv(
    "fig2_decode_regimes_context.csv",
    ["context_tokens", "weight_GB_per_step", "KV_GB_per_step", "total_GB_per_step"],
    rows_cross,
)

# -----------------------------------------------------------------------------
# Cleaner exact published-evidence figure: KIVI and DistServe
# -----------------------------------------------------------------------------
fig = plt.figure(figsize=(7.15, 2.38))
ax1 = fig.add_axes([0.125, 0.22, 0.33, 0.67])
ax2 = fig.add_axes([0.565, 0.22, 0.385, 0.67])

# KIVI range/points
metrics = ["Peak-memory\nreduction", "Maximum batch\nsize", "Throughput\nspeedup"]
lows = [2.6, 4.0, 2.35]
highs = [2.6, 4.0, 3.47]
y = np.arange(3)[::-1]
for yi, lo, hi in zip(y, lows, highs):
    if abs(hi-lo) < 1e-9:
        ax1.scatter([lo], [yi], s=34, color=NAVY, zorder=4)
        ax1.text(lo+0.10, yi, f"{lo:.2f}×", va="center", fontsize=6.2, color=INK)
    else:
        ax1.plot([lo, hi], [yi, yi], color=NAVY, lw=6, solid_capstyle="round")
        ax1.scatter([lo, hi], [yi, yi], s=20, color=NAVY, zorder=4)
        ax1.text(hi+0.10, yi, f"{lo:.2f}–{hi:.2f}×", va="center", fontsize=6.2, color=INK)
ax1.set_yticks(y, metrics)
ax1.set_xlim(0, 4.7)
ax1.set_xlabel("Reported factor (×)")
ax1.set_title("(a) KIVI 2-bit KV-cache results", loc="left", fontweight="bold")
ax1.grid(axis="x", color=GRID)
ax1.spines[["top", "right", "left"]].set_visible(False)
ax1.tick_params(axis="y", length=0)

# DistServe grouped bars
baselines = ["vLLM", "DeepSpeed-MII"]
req = [4.3, 1.8]
slo = [12.6, 2.6]
x = np.arange(len(baselines))
w = 0.34
bars1 = ax2.bar(x-w/2, req, width=w, color=PALE_BLUE, edgecolor=NAVY, lw=1.0, hatch="//", label="Higher request rate")
bars2 = ax2.bar(x+w/2, slo, width=w, color=PALE_ORANGE, edgecolor=ORANGE, lw=1.0, hatch="xx", label="Tighter SLO")
for bars in [bars1, bars2]:
    for b in bars:
        ax2.text(b.get_x()+b.get_width()/2, b.get_height()+0.22, f"{b.get_height():.1f}×",
                 ha="center", va="bottom", fontsize=6.0, color=INK)
ax2.set_xticks(x, baselines)
ax2.set_ylim(0, 14.2)
ax2.set_ylabel("Improvement factor (×)")
ax2.set_title("(b) DistServe, OPT-66B summarization", loc="left", fontweight="bold")
ax2.grid(axis="y", color=GRID)
ax2.spines[["top", "right"]].set_visible(False)
ax2.legend(loc="upper right", frameon=True)

fig.text(0.51, 0.04,
         "Exact values reported by the cited papers; metrics, baselines, and workloads are not interchangeable.",
         ha="center", va="bottom", fontsize=5.7, color=MUTED)
save_pair(fig, "fig8_quant_disagg_evidence")

# -----------------------------------------------------------------------------
# Reference architecture: explanatory redraw, not a measured topology
# -----------------------------------------------------------------------------
fig = plt.figure(figsize=(7.15, 3.02))
ax = fig.add_axes([0.018, 0.06, 0.964, 0.90])
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")

# Horizontal bands
ax.add_patch(Rectangle((0.0, 0.59), 1.0, 0.36, fc="#F7F9FB", ec="none", zorder=0))
ax.add_patch(Rectangle((0.0, 0.29), 1.0, 0.25, fc="#FBFBFC", ec="none", zorder=0))
ax.add_patch(Rectangle((0.0, 0.04), 1.0, 0.18, fc="#F7F9FB", ec="none", zorder=0))
ax.text(0.008, 0.77, "DATA PLANE", rotation=90, fontsize=5.5, color=MUTED, va="center")
ax.text(0.008, 0.415, "CONTROL PLANE", rotation=90, fontsize=5.5, color=MUTED, va="center")
ax.text(0.008, 0.13, "TELEMETRY", rotation=90, fontsize=5.5, color=MUTED, va="center")

# Data plane boxes
box(ax, (0.03, 0.69), (0.10, 0.14), "Ingress", "tier · adapter\npolicy version", fc=PALE_GRAY)
box(ax, (0.17, 0.64), (0.19, 0.24), "Prefill pool", "compute-oriented\nlarge prompt batches", fc=PALE_GOLD, ec=GOLD, title_color=GOLD, title_size=7.1)
ax.add_patch(FancyBboxPatch((0.40, 0.62), 0.19, 0.28,
                             boxstyle="round,pad=0.006,rounding_size=0.014",
                             fc=PALE_TEAL, ec=TEAL, lw=0.9, zorder=3))
ax.text(0.495, 0.855, "KV state fabric", ha="center", va="center",
        fontsize=7.1, fontweight="bold", color=TEAL, zorder=5)
ax.text(0.495, 0.815, "paged · refcounted · versioned", ha="center", va="center",
        fontsize=4.8, color=MUTED, zorder=5)
box(ax, (0.63, 0.64), (0.19, 0.24), "Decode pool", "bandwidth/capacity\nlarge continuous batch", fc=PALE_BLUE, ec=NAVY, title_color=NAVY, title_size=7.1)
box(ax, (0.86, 0.69), (0.11, 0.14), "Egress", "token stream\nSLO accounting", fc=PALE_GRAY)

# Fabric tiers inside KV box
for i, (label, fc, y) in enumerate([
    ("HBM", "#CFE4EF", 0.745),
    ("Host DRAM", "#9EC9DA", 0.695),
    ("NVMe / remote", "#6BA8C0", 0.645),
]):
    ax.add_patch(Rectangle((0.425, y), 0.14, 0.035, fc=fc, ec=NAVY, lw=0.45, zorder=5))
    ax.text(0.495, y+0.0175, label, ha="center", va="center", fontsize=4.9, color=INK, zorder=6)

arrow(ax, (0.13, 0.76), (0.17, 0.76), color=INK, lw=1.4)
arrow(ax, (0.36, 0.76), (0.40, 0.76), color=TEAL, lw=1.8)
arrow(ax, (0.59, 0.76), (0.63, 0.76), color=TEAL, lw=1.8)
arrow(ax, (0.82, 0.76), (0.86, 0.76), color=INK, lw=1.4)
ax.text(0.15, 0.79, "prompt", fontsize=5.2, color=MUTED, ha="center")
ax.text(0.38, 0.79, "KV write", fontsize=5.2, color=TEAL, ha="center")
ax.text(0.61, 0.79, "KV read", fontsize=5.2, color=TEAL, ha="center")
ax.text(0.84, 0.79, "tokens", fontsize=5.2, color=MUTED, ha="center")

# Control bus
ax.plot([0.045, 0.955], [0.565, 0.565], color=INK, lw=1.1, zorder=2)
ax.text(0.055, 0.58, "control bus", fontsize=5.1, color=MUTED)

controls = [
    (0.035, "KV directory", "ownership · refcount\nprecision · tier"),
    (0.225, "Global scheduler", "admission · slack\nchunk size · pool ratio"),
    (0.415, "Speculation", "draft source · depth\nacceptance by regime"),
    (0.605, "MoE placement", "expert replication\nall-to-all schedule"),
    (0.795, "RL coordinator", "policy version\ntrajectory metadata"),
]
for x, title, subtitle in controls:
    box(ax, (x, 0.34), (0.17, 0.145), title, subtitle, fc=WHITE, ec=INK,
        title_size=6.2, subtitle_size=4.8)
    ax.plot([x+0.085, x+0.085], [0.485, 0.565], color=INK, lw=0.75, zorder=2)

# Selected control arrows to data plane
for x, target, col in [
    (0.31, (0.265, 0.64), GOLD),
    (0.31, (0.725, 0.64), NAVY),
    (0.50, (0.725, 0.64), NAVY),
    (0.69, (0.725, 0.64), ORANGE),
    (0.88, (0.725, 0.64), NAVY),
]:
    arrow(ax, (x, 0.565), target, color=col, lw=0.85, ls="--", ms=6, z=2.5)

# Telemetry band
box(ax, (0.055, 0.075), (0.89, 0.10), "Observability and online cost model",
    "bytes by hierarchy boundary · KV occupancy · memory/network stalls · TTFT/TPOT tails · goodput · joules per useful token/trajectory",
    fc=WHITE, ec=INK, title_size=6.6, subtitle_size=4.9, title_y=0.65)
arrow(ax, (0.50, 0.22), (0.50, 0.29), color=MUTED, lw=0.85, ls="--", ms=6)
ax.text(0.515, 0.255, "operating point", fontsize=4.9, color=MUTED, va="center")
for x in [0.12, 0.31, 0.50, 0.69, 0.88]:
    arrow(ax, (x, 0.34), (x, 0.175), color=MUTED, lw=0.65, ls=":", ms=5)

# Legend
ax.plot([0.055, 0.085], [0.025, 0.025], color=INK, lw=1.3)
ax.text(0.093, 0.025, "tokens / state", fontsize=4.9, color=MUTED, va="center")
ax.plot([0.205, 0.235], [0.025, 0.025], color=INK, lw=0.9, ls="--")
ax.text(0.243, 0.025, "control", fontsize=4.9, color=MUTED, va="center")
ax.add_patch(Rectangle((0.35, 0.012), 0.018, 0.026, fc=PALE_GOLD, ec=GOLD, lw=0.6))
ax.text(0.375, 0.025, "compute-oriented", fontsize=4.9, color=MUTED, va="center")
ax.add_patch(Rectangle((0.49, 0.012), 0.018, 0.026, fc=PALE_BLUE, ec=NAVY, lw=0.6))
ax.text(0.515, 0.025, "bandwidth-oriented", fontsize=4.9, color=MUTED, va="center")
ax.add_patch(Rectangle((0.66, 0.012), 0.018, 0.026, fc=PALE_TEAL, ec=TEAL, lw=0.6))
ax.text(0.685, 0.025, "state / transport", fontsize=4.9, color=MUTED, va="center")

save_pair(fig, "fig12_reference_architecture")

print("Generated integrated figures in", FIG_DIR)
