from __future__ import annotations

from pathlib import Path
import csv

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

# Print-safe palette. Meaning is also encoded with line style and labels.
INK = "#17212B"
MUTED = "#64707C"
GRID = "#D9E0E6"
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
        "font.size": 7.2,
        "axes.titlesize": 8.0,
        "axes.labelsize": 7.0,
        "xtick.labelsize": 6.4,
        "ytick.labelsize": 6.4,
        "legend.fontsize": 6.3,
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


def box(ax, xy, wh, title, subtitle=None, *, fc=WHITE, ec=INK,
        title_color=INK, lw=0.9, title_size=7.0, subtitle_size=5.55,
        radius=0.016, z=3, title_y=0.62):
    x, y = xy
    w, h = wh
    p = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.006,rounding_size={radius}",
        fc=fc, ec=ec, lw=lw, zorder=z,
    )
    ax.add_patch(p)
    if subtitle is None:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center",
                fontsize=title_size, fontweight="bold", color=title_color, zorder=z + 1)
    else:
        ax.text(x + w / 2, y + h * title_y, title, ha="center", va="center",
                fontsize=title_size, fontweight="bold", color=title_color, zorder=z + 1)
        ax.text(x + w / 2, y + h * 0.28, subtitle, ha="center", va="center",
                fontsize=subtitle_size, color=MUTED, linespacing=1.15, zorder=z + 1)
    return p


def arrow(ax, start, end, *, color=INK, lw=1.15, ls="-", rad=0.0,
          ms=9, z=4, arrowstyle="-|>"):
    p = FancyArrowPatch(
        start, end, arrowstyle=arrowstyle, mutation_scale=ms,
        color=color, lw=lw, linestyle=ls,
        connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0,
        zorder=z,
    )
    ax.add_patch(p)
    return p


def label(ax, x, y, text, *, color=MUTED, size=5.5, weight="normal",
          ha="center", va="center", rotation=0, fc=WHITE, pad=0.10, z=7):
    ax.text(x, y, text, fontsize=size, color=color, fontweight=weight,
            ha=ha, va=va, rotation=rotation, zorder=z,
            bbox=dict(boxstyle=f"round,pad={pad}", fc=fc, ec="none", alpha=0.96))


def worker_pool(ax, x, y, w, h, *, title, subtitle, edge, fill, prefix):
    """Clean pool with a header and three worker cards."""
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.006,rounding_size=0.016",
        fc=fill, ec=edge, lw=1.05, zorder=2,
    ))
    ax.add_patch(Rectangle((x, y + h * 0.75), w, h * 0.25, fc=edge, ec="none", zorder=3))
    ax.text(x + w / 2, y + h * 0.875, title, ha="center", va="center",
            fontsize=7.0, fontweight="bold", color=WHITE, zorder=4)
    ax.text(x + w / 2, y + h * 0.665, subtitle, ha="center", va="center",
            fontsize=5.55, color=INK, zorder=4)
    gap = w * 0.045
    card_w = (w - 4 * gap) / 3
    card_y = y + h * 0.17
    card_h = h * 0.34
    for i in range(3):
        cx = x + gap + i * (card_w + gap)
        ax.add_patch(FancyBboxPatch(
            (cx, card_y), card_w, card_h,
            boxstyle="round,pad=0.003,rounding_size=0.008",
            fc=WHITE, ec=edge, lw=0.65, zorder=4,
        ))
        ax.text(cx + card_w / 2, card_y + card_h * 0.62, f"{prefix}{i}",
                ha="center", va="center", fontsize=6.1, fontweight="bold",
                color=edge, zorder=5)
        ax.text(cx + card_w / 2, card_y + card_h * 0.27, "worker",
                ha="center", va="center", fontsize=4.7, color=MUTED, zorder=5)


def panel_title(ax, text):
    ax.text(0.0, 0.985, text, transform=ax.transAxes, ha="left", va="top",
            fontsize=8.1, fontweight="bold", color=INK)


# =============================================================================
# FIGURE 7 — Prefill/decode disaggregation
# =============================================================================
fig = plt.figure(figsize=(7.15, 2.63))
ax = fig.add_axes([0.025, 0.07, 0.95, 0.88])
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")

# Control plane
box(ax, (0.31, 0.82), (0.38, 0.12), "Global admission and placement controller",
    "independent queues, autoscaling, and SLO-aware routing",
    fc=PALE_GRAY, title_size=6.9, subtitle_size=5.25, title_y=0.64)

# Data-plane objects
box(ax, (0.005, 0.40), (0.105, 0.20), "Request queue", "prompt tokens",
    fc=PALE_GRAY, title_size=6.7, subtitle_size=5.3)
worker_pool(ax, 0.16, 0.31, 0.245, 0.37,
            title="PREFILL POOL", subtitle="compute-oriented · TTFT",
            edge=NAVY, fill=PALE_BLUE, prefix="P")
box(ax, (0.455, 0.36), (0.115, 0.27), "KV handoff",
    "layer-wise blocks\nRDMA / NVLink",
    fc=PALE_TEAL, ec=TEAL, title_color=TEAL,
    title_size=6.45, subtitle_size=5.1, title_y=0.67)
worker_pool(ax, 0.62, 0.31, 0.245, 0.37,
            title="DECODE POOL", subtitle="bandwidth/capacity · TPOT",
            edge=ORANGE, fill=PALE_ORANGE, prefix="D")
box(ax, (0.915, 0.40), (0.080, 0.20), "Token\nstream", "first + next",
    fc=PALE_GRAY, title_size=6.25, subtitle_size=4.9, title_y=0.66)

# Control arrows behind the data paths
arrow(ax, (0.40, 0.82), (0.285, 0.68), color=NAVY, lw=1.0, ls="--", rad=0.05, z=2.5)
arrow(ax, (0.60, 0.82), (0.742, 0.68), color=ORANGE, lw=1.0, ls="--", rad=-0.05, z=2.5)
label(ax, 0.332, 0.755, "place / scale", color=NAVY, size=5.15)
label(ax, 0.670, 0.755, "place / scale", color=ORANGE, size=5.15)
arrow(ax, (0.110, 0.58), (0.335, 0.82), color=INK, lw=0.9, ls="--", rad=-0.08, z=2.4)
label(ax, 0.195, 0.715, "arrival metadata", color=MUTED, size=5.0, rotation=26)

# Request, KV, and token flows
arrow(ax, (0.110, 0.49), (0.160, 0.49), color=INK, lw=1.45)
label(ax, 0.135, 0.535, "prompt", color=INK, size=5.15)

arrow(ax, (0.405, 0.49), (0.455, 0.49), color=TEAL, lw=2.15)
label(ax, 0.430, 0.447, "KV blocks", color=TEAL, size=5.20, weight="bold")
arrow(ax, (0.570, 0.49), (0.620, 0.49), color=TEAL, lw=2.15)
label(ax, 0.595, 0.447, "pull / stream", color=TEAL, size=5.05)

# Prefill emits the first token while decode emits subsequent tokens.
# Route the first-token lane above both pools to avoid crossing headers and labels.
ax.plot([0.405, 0.445, 0.875], [0.635, 0.715, 0.715],
        color=NAVY, lw=1.25, solid_capstyle="round", zorder=5)
arrow(ax, (0.875, 0.715), (0.915, 0.585), color=NAVY, lw=1.25, rad=0.0, z=5)
label(ax, 0.665, 0.715, "first token", color=NAVY, size=5.35, weight="bold")
arrow(ax, (0.865, 0.45), (0.915, 0.45), color=ORANGE, lw=1.55)
label(ax, 0.895, 0.485, "next tokens", color=ORANGE, size=5.05, weight="bold")

# Independent provisioning cards
box(ax, (0.16, 0.085), (0.245, 0.125), "Prefill provisioning",
    "prompt batching · tensor/pipeline parallelism · FLOP density",
    fc=PALE_BLUE, ec=NAVY, title_color=NAVY,
    title_size=6.1, subtitle_size=4.95, title_y=0.65)
box(ax, (0.62, 0.085), (0.245, 0.125), "Decode provisioning",
    "large live batch · replication · HBM capacity/bandwidth",
    fc=PALE_ORANGE, ec=ORANGE, title_color=ORANGE,
    title_size=6.1, subtitle_size=4.95, title_y=0.65)

# Legend and one-time transfer note
ax.plot([0.012, 0.040], [0.145, 0.145], color=INK, lw=1.45)
ax.text(0.047, 0.145, "tokens / state", fontsize=5.25, color=MUTED, va="center")
ax.plot([0.012, 0.040], [0.095, 0.095], color=INK, lw=0.95, ls="--")
ax.text(0.047, 0.095, "control", fontsize=5.25, color=MUTED, va="center")
ax.text(0.5125, 0.255, "one-time KV handoff; overlap transfer with prefill when possible",
        fontsize=5.25, color=MUTED, ha="center", va="center")

save_pair(fig, "fig8_pd_disaggregation")


# =============================================================================
# FIGURE 8 — MegaScale-Infer conceptual architecture + exact reported metrics
# =============================================================================
fig = plt.figure(figsize=(7.15, 2.46))
axL = fig.add_axes([0.035, 0.10, 0.565, 0.84])
axL.set_xlim(0, 1)
axL.set_ylim(0, 1)
axL.axis("off")
panel_title(axL, "(a) Disaggregated attention and expert execution")

# Attention and expert groups
axL.add_patch(FancyBboxPatch((0.02, 0.52), 0.40, 0.34,
                             boxstyle="round,pad=0.006,rounding_size=0.016",
                             fc=PALE_BLUE, ec=NAVY, lw=0.95))
axL.add_patch(Rectangle((0.02, 0.78), 0.40, 0.08, fc=NAVY, ec="none"))
axL.text(0.22, 0.82, "ATTENTION REPLICAS (M)", ha="center", va="center",
         fontsize=6.6, fontweight="bold", color=WHITE)
axL.text(0.22, 0.735, "attention weights + KV cache", ha="center", va="center",
         fontsize=5.35, color=MUTED)

axL.add_patch(FancyBboxPatch((0.61, 0.52), 0.37, 0.34,
                             boxstyle="round,pad=0.006,rounding_size=0.016",
                             fc=PALE_ORANGE, ec=ORANGE, lw=0.95))
axL.add_patch(Rectangle((0.61, 0.78), 0.37, 0.08, fc=ORANGE, ec="none"))
axL.text(0.795, 0.82, "EXPERT GROUP (N)", ha="center", va="center",
         fontsize=6.6, fontweight="bold", color=WHITE)
axL.text(0.795, 0.735, "expert-parallel FFN parameters", ha="center", va="center",
         fontsize=5.35, color=MUTED)

# Worker cards
for i, x in enumerate([0.05, 0.17, 0.29]):
    box(axL, (x, 0.575), (0.095, 0.125), f"A{i}", "TP replica",
        fc=WHITE, ec=NAVY, title_color=NAVY,
        title_size=6.4, subtitle_size=4.65, radius=0.010, title_y=0.64)
for i, x in enumerate([0.635, 0.715, 0.795, 0.875]):
    box(axL, (x, 0.575), (0.065, 0.125), f"E{i}", "TP",
        fc=WHITE, ec=ORANGE, title_color=ORANGE,
        title_size=6.15, subtitle_size=4.55, radius=0.009, title_y=0.64)

# Dispatch/combine paths are separated vertically for unambiguous direction.
arrow(axL, (0.42, 0.695), (0.61, 0.695), color=TEAL, lw=1.9)
label(axL, 0.515, 0.735, "M2N dispatch", color=TEAL, size=5.5, weight="bold")
arrow(axL, (0.61, 0.585), (0.42, 0.585), color=GOLD, lw=1.9)
label(axL, 0.515, 0.545, "N2M combine", color=GOLD, size=5.5, weight="bold")

# Timeline band
axL.text(0.02, 0.445, "Ping-pong micro-batches hide communication behind compute",
         fontsize=6.05, fontweight="bold", color=INK, ha="left")
row_x0, row_x1 = 0.13, 0.965
axL.text(0.105, 0.337, "Attention", fontsize=5.25, color=NAVY, ha="right", va="center")
axL.text(0.105, 0.215, "Experts", fontsize=5.25, color=ORANGE, ha="right", va="center")
axL.plot([row_x0, row_x1], [0.337, 0.337], color=GRID, lw=0.65)
axL.plot([row_x0, row_x1], [0.215, 0.215], color=GRID, lw=0.65)

w = 0.18
attn_starts = [0.14, 0.42, 0.70]
expert_starts = [0.28, 0.56, 0.78]
for i, (sa, se) in enumerate(zip(attn_starts, expert_starts), start=1):
    axL.add_patch(FancyBboxPatch((sa, 0.296), w, 0.082,
                                 boxstyle="round,pad=0.002,rounding_size=0.006",
                                 fc=PALE_BLUE, ec=NAVY, lw=0.65))
    axL.text(sa + w / 2, 0.337, rf"$\mu_{i}$", ha="center", va="center",
             fontsize=6.0, fontweight="bold", color=NAVY)
    ew = min(w, 0.975 - se)
    axL.add_patch(FancyBboxPatch((se, 0.174), ew, 0.082,
                                 boxstyle="round,pad=0.002,rounding_size=0.006",
                                 fc=PALE_ORANGE, ec=ORANGE, lw=0.65))
    axL.text(se + ew / 2, 0.215, rf"$\mu_{i}$", ha="center", va="center",
             fontsize=6.0, fontweight="bold", color=ORANGE)

arrow(axL, (0.20, 0.115), (0.91, 0.115), color=MUTED, lw=0.75,
      arrowstyle="<->", ms=8)
axL.text(0.555, 0.073, "overlap window", ha="center", va="center",
         fontsize=5.25, color=MUTED)

# Right-side exact reported metrics
axR = fig.add_axes([0.685, 0.47, 0.285, 0.39])
axR.set_title("(b) Reported per-GPU decode gain", loc="left",
              fontsize=7.55, fontweight="bold", pad=7, color=INK)
labels = ["vLLM", "TensorRT-\nLLM"]
values = np.array([7.11, 1.90])
y = np.arange(2)
bars = axR.barh(y, values, height=0.46, color=[NAVY, BLUE], edgecolor=INK, linewidth=0.55)
axR.set_yticks(y, labels)
axR.invert_yaxis()
axR.set_xlim(0, 8.15)
axR.set_xticks([0, 2, 4, 6, 8])
axR.set_xlabel("MegaScale-Infer / baseline (×)", labelpad=2)
axR.grid(axis="x", color=GRID)
axR.set_axisbelow(True)
axR.spines[["top", "right"]].set_visible(False)
for b, v in zip(bars, values):
    axR.text(v + 0.12, b.get_y() + b.get_height() / 2, f"{v:.2f}×",
             ha="left", va="center", fontsize=6.6, fontweight="bold", color=INK)

axC = fig.add_axes([0.685, 0.11, 0.285, 0.245])
axC.set_xlim(0, 1)
axC.set_ylim(0, 1)
axC.axis("off")
axC.text(0.0, 0.97, "M2N communication at 256-KB messages",
         fontsize=6.15, fontweight="bold", color=INK, ha="left", va="top")
for x, value, unit, desc, ec, fc in [
    (0.00, "4.2×", "throughput", "higher than NCCL", TEAL, PALE_TEAL),
    (0.52, "68.2%", "median latency", "lower than NCCL", GOLD, PALE_GOLD),
]:
    axC.add_patch(FancyBboxPatch((x, 0.12), 0.46, 0.67,
                                 boxstyle="round,pad=0.008,rounding_size=0.025",
                                 fc=fc, ec=ec, lw=0.85))
    axC.text(x + 0.23, 0.57, value, ha="center", va="center",
             fontsize=10.6, fontweight="bold", color=ec)
    axC.text(x + 0.23, 0.35, unit, ha="center", va="center",
             fontsize=5.6, fontweight="bold", color=INK)
    axC.text(x + 0.23, 0.20, desc, ha="center", va="center",
             fontsize=5.1, color=MUTED)

save_pair(fig, "fig10_moe_serving")


# =============================================================================
# FIGURE 9 — Asynchronous RL loop + exact OpenRLHF measurements
# =============================================================================
fig = plt.figure(figsize=(7.15, 2.615))
axL = fig.add_axes([0.035, 0.11, 0.565, 0.84])
axL.set_xlim(0, 1)
axL.set_ylim(0, 1)
axL.axis("off")
panel_title(axL, "(a) Asynchronous RL post-training dataflow")

# Top row
box(axL, (0.02, 0.62), (0.19, 0.18), "Prompt / task\nsampler", "sampling + curriculum",
    fc=PALE_GRAY, title_size=6.05, subtitle_size=4.85, title_y=0.67)
box(axL, (0.29, 0.60), (0.25, 0.22), "Rollout inference\nfleet",
    "continuous batching\npaged KV + resumable state",
    fc=PALE_BLUE, ec=NAVY, title_color=NAVY,
    title_size=6.10, subtitle_size=4.70, title_y=0.72)
box(axL, (0.65, 0.60), (0.22, 0.22), "Tools /\nenvironment",
    "tool calls or simulators\nvariable-latency steps",
    fc=PALE_TEAL, ec=TEAL, title_color=TEAL,
    title_size=6.00, subtitle_size=4.65, title_y=0.72)

# Bottom row
box(axL, (0.65, 0.20), (0.22, 0.22), "Trajectory\nbuffer",
    "versions + log-probs\nstraggler-tolerant queue",
    fc=PALE_GRAY, title_size=6.00, subtitle_size=4.65, title_y=0.72)
box(axL, (0.29, 0.20), (0.25, 0.22), "Policy trainer",
    "policy update\nreshard + broadcast",
    fc=PALE_ORANGE, ec=ORANGE, title_color=ORANGE,
    title_size=6.15, subtitle_size=4.75, title_y=0.70)
box(axL, (0.02, 0.20), (0.19, 0.22), "Parameter\nservice",
    "versioned actor weights",
    fc=PALE_GOLD, ec=GOLD, title_color=GOLD,
    title_size=6.00, subtitle_size=4.75, title_y=0.69)

# Solid experience path; labels sit in clear inter-node gaps.
arrow(axL, (0.21, 0.71), (0.29, 0.71), color=INK, lw=1.35)
label(axL, 0.25, 0.755, "prompts", color=INK, size=5.1)
arrow(axL, (0.54, 0.745), (0.65, 0.745), color=TEAL, lw=1.45)
label(axL, 0.595, 0.785, "actions", color=TEAL, size=5.1, weight="bold")
arrow(axL, (0.65, 0.665), (0.54, 0.665), color=TEAL, lw=1.05, rad=-0.11)
label(axL, 0.595, 0.625, "observations", color=TEAL, size=5.0)
arrow(axL, (0.76, 0.60), (0.76, 0.42), color=INK, lw=1.35)
label(axL, 0.79, 0.51, "completed\ntrajectories", color=MUTED, size=5.05, ha="left")
arrow(axL, (0.65, 0.31), (0.54, 0.31), color=INK, lw=1.35)
label(axL, 0.595, 0.355, "training batch", color=INK, size=5.0)

# Dashed weight/control path. Route feedback along the clear lower/left margin.
arrow(axL, (0.29, 0.31), (0.21, 0.31), color=ORANGE, lw=1.15, ls="--")
label(axL, 0.25, 0.355, "new policy", color=ORANGE, size=5.0, weight="bold")
arrow(axL, (0.115, 0.42), (0.365, 0.60), color=GOLD, lw=1.2, ls="--", rad=-0.14)
label(axL, 0.195, 0.525, r"weights $\theta_k$", color=GOLD,
      size=5.05, weight="bold", rotation=28)

# Concise legend and correctness note
axL.plot([0.02, 0.055], [0.095, 0.095], color=INK, lw=1.4)
axL.text(0.062, 0.095, "trajectory / experience", fontsize=5.15, color=MUTED, va="center")
axL.plot([0.36, 0.395], [0.095, 0.095], color=INK, lw=1.05, ls="--")
axL.text(0.402, 0.095, "weights / control", fontsize=5.15, color=MUTED, va="center")
axL.text(0.02, 0.035,
         "Each trajectory carries its policy version, rollout log-probabilities, and resumable state.",
         fontsize=5.2, color=MUTED, ha="left")

# Exact reported comparison
rows = []
with (DATA_DIR / "fig9_openrlhf_14b_step_times.csv").open(newline="") as f:
    rows = list(csv.DictReader(f))
lengths = np.array([int(r["max_generation_K_tokens"]) for r in rows])
openrlhf = np.array([float(r["OpenRLHF_seconds_per_step"]) for r in rows])
verl = np.array([float(r["verl_seconds_per_step"]) for r in rows])
ratios = verl / openrlhf
x = np.arange(len(lengths))

axR = fig.add_axes([0.665, 0.30, 0.305, 0.56])
axR.set_title("(b) 14B long-CoT RLVR step time", loc="left",
              fontsize=7.8, fontweight="bold", pad=7, color=INK)
axR.plot(x, openrlhf, color=NAVY, marker="o", markersize=4.0, lw=1.55, label="OpenRLHF")
axR.plot(x, verl, color=ORANGE, marker="s", markersize=3.8, lw=1.45, ls="--", label="verl")
axR.fill_between(x, openrlhf, verl, color=PALE_GOLD, alpha=0.95, zorder=0)
axR.set_xticks(x, [f"{v}K" for v in lengths])
axR.set_ylim(0, 555)
axR.set_yticks([0, 100, 200, 300, 400, 500])
axR.set_ylabel("Average step time (s)")
axR.set_xlabel("Maximum generation length")
axR.grid(axis="y", color=GRID)
axR.set_axisbelow(True)
axR.spines[["top", "right"]].set_visible(False)
axR.legend(frameon=False, loc="upper left", ncol=1, handlelength=2.0)
axR.text(0.04, 0.72, "8 × H200 GPUs", transform=axR.transAxes,
         ha="left", va="top", fontsize=5.55, color=MUTED)

# Values are placed consistently: OpenRLHF below, verl above.
for i, (xi, y1, y2) in enumerate(zip(x, openrlhf, verl)):
    lower_offset = -5 if i == 0 else -9
    axR.annotate(f"{y1:.1f}", (xi, y1), xytext=(0, lower_offset), textcoords="offset points",
                 ha="center", va="bottom", fontsize=5.25, color=NAVY)
    axR.annotate(f"{y2:.1f}", (xi, y2), xytext=(0, 6), textcoords="offset points",
                 ha="center", va="bottom", fontsize=5.25, color=ORANGE)

ratio_text = "verl / OpenRLHF:  " + "   ·   ".join(
    f"{L}K {r:.2f}×" for L, r in zip(lengths, ratios)
)
fig.text(0.8175, 0.155, ratio_text, ha="center", va="center",
         fontsize=5.05, color=MUTED)
fig.text(0.8175, 0.080, "Exact Table 1 values; first 10 steps excluded.",
         ha="center", va="center", fontsize=5.05, color=MUTED)

save_pair(fig, "fig9_rl_loop")

print("Recreated manuscript Figures 7, 8, and 9 as vector PDF + 360-dpi PNG.")
