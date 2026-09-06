from __future__ import annotations

import csv
import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch

ROOT = Path(__file__).resolve().parent
FIG = ROOT / "figures"
DATA = ROOT / "data"
FIG.mkdir(exist_ok=True)
DATA.mkdir(exist_ok=True)

mpl.rcParams.update({
    "font.family": "serif",
    "font.size": 8,
    "axes.titlesize": 8.5,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 6.7,
    "figure.dpi": 180,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.linewidth": 0.65,
    "grid.linewidth": 0.45,
})


def save(fig, name: str):
    fig.savefig(FIG / name, format="pdf", bbox_inches="tight")
    plt.close(fig)


def write_csv(name: str, rows: list[dict]):
    if not rows:
        return
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with open(DATA / name, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def box(ax, xy, w, h, text, fc="white", lw=0.8, fontsize=7, radius=0.03):
    p = FancyBboxPatch(xy, w, h, boxstyle=f"round,pad=0.015,rounding_size={radius}",
                       ec="black", fc=fc, lw=lw)
    ax.add_patch(p)
    ax.text(xy[0]+w/2, xy[1]+h/2, text, ha="center", va="center", fontsize=fontsize)
    return p


def arrow(ax, p1, p2, text=None, fontsize=6.5, style="-|>", connectionstyle="arc3"):
    a = FancyArrowPatch(p1, p2, arrowstyle=style, mutation_scale=8, lw=0.8,
                        color="black", connectionstyle=connectionstyle)
    ax.add_patch(a)
    if text:
        ax.text((p1[0]+p2[0])/2, (p1[1]+p2[1])/2+0.018, text, ha="center", va="bottom", fontsize=fontsize)
    return a


# Figure 1: exact historical values from Horowitz (45 nm table)
energy_rows = [
    {"operation": "32-bit int add", "energy_pJ": 0.1},
    {"operation": "32-bit FP add", "energy_pJ": 0.9},
    {"operation": "32-bit register-file access", "energy_pJ": 1.0},
    {"operation": "32-bit int multiply", "energy_pJ": 3.1},
    {"operation": "32-bit FP multiply", "energy_pJ": 3.7},
    {"operation": "32-bit 8-KB SRAM read", "energy_pJ": 5.0},
    {"operation": "32-bit DRAM read", "energy_pJ": 640.0},
]
write_csv("fig1_horowitz_energy_45nm.csv", energy_rows)
fig, ax = plt.subplots(figsize=(3.55, 2.45))
labels = [r["operation"].replace("32-bit ", "") for r in energy_rows]
vals = [r["energy_pJ"] for r in energy_rows]
x = np.arange(len(vals))
bars = ax.bar(x, vals, edgecolor="black", facecolor="0.85", hatch=["", "//", "..", "xx", "\\\\", "--", "++"])
ax.set_yscale("log")
ax.set_ylabel("Energy per operation/access (pJ, log scale)")
ax.set_xticks(x, labels, rotation=35, ha="right")
ax.set_ylim(0.07, 1300)
ax.grid(axis="y", which="both", alpha=0.35)
for b, v in zip(bars, vals):
    ax.text(b.get_x()+b.get_width()/2, v*1.18, f"{v:g}", ha="center", va="bottom", fontsize=6.7)
ax.set_title("Published 45-nm energy hierarchy")
ax.text(0.01, -0.44, "Source values: Horowitz, ISSCC 2014. Historical estimates, not modern-chip specifications.",
        transform=ax.transAxes, fontsize=6.2, va="top")
save(fig, "fig1_energy_hierarchy.pdf")


# Figure 2: derived traffic using stated Llama-2-70B-style configuration
P = 70e9
bw = 2.0
L = 80
Hkv = 8
dh = 128
bkv = 2.0
S = 8192
Ktok = 2*L*Hkv*dh*bkv
B = np.array([1,2,4,8,16,32,52,64,128,256])
weight_gb = P*bw/B/1e9
kv_gb = np.full_like(B, S*Ktok/1e9, dtype=float)
rows=[]
for b,w,k in zip(B,weight_gb,kv_gb): rows.append({"batch_size":int(b),"weight_GB_per_output_token":w,"KV_GB_per_output_token":k})
write_csv("fig2_derived_traffic_llama2_70b.csv", rows)
fig, ax = plt.subplots(figsize=(3.55, 2.35))
ax.plot(B, weight_gb, marker="o", ms=3.5, lw=1.1, label="Amortized weight read")
ax.plot(B, kv_gb, marker="s", ms=3.2, lw=1.1, label="Private KV read (8K context)")
ax.axvline(P*bw/(S*Ktok), ls="--", lw=0.8, color="0.25")
ax.text(52, max(kv_gb)*1.15, r"$B_{\times}\approx52$", ha="center", fontsize=7)
ax.set_xscale("log", base=2); ax.set_yscale("log")
ax.set_xlabel("Decode batch size")
ax.set_ylabel("Read traffic per output token (GB)")
ax.set_xticks(B, [str(int(v)) for v in B])
ax.grid(which="both", alpha=0.3)
ax.legend(frameon=False, loc="upper right")
ax.set_title("Derived weight/KV traffic crossover")
ax.text(0.01, -0.31, "Parameters: P=70B, L=80, Hkv=8, dh=128, BF16 weights/KV, context=8192.",
        transform=ax.transAxes, fontsize=6.2, va="top")
save(fig, "fig2_traffic_breakdown.pdf")


# Figure 3: derived KV capacity exact formula
contexts = np.array([2048, 8192, 32768, 131072])
configs = [
    ("MHA, BF16 (64 KV heads)", 64, 2.0),
    ("GQA, BF16 (8 KV heads)", 8, 2.0),
    ("GQA, INT8 KV", 8, 1.0),
    ("GQA, 2-bit KV", 8, 0.25),
]
rows=[]
fig, ax = plt.subplots(figsize=(3.55, 2.35))
markers=["o","s","^","D"]
for (label,h,b),m in zip(configs,markers):
    cap = contexts*(2*L*h*dh*b)/(1024**3)
    ax.plot(contexts/1024, cap, marker=m, ms=3.4, lw=1.1, label=label)
    for c,v in zip(contexts,cap): rows.append({"configuration":label,"context_tokens":int(c),"KV_GiB_per_sequence":v})
write_csv("fig3_derived_kv_capacity.csv", rows)
ax.set_xscale("log", base=4); ax.set_yscale("log")
ax.set_xlabel("Context length (K tokens)"); ax.set_ylabel("KV capacity per sequence (GiB)")
ax.set_xticks(contexts/1024, ["2","8","32","128"])
ax.grid(which="both", alpha=0.3)
ax.legend(frameon=False, loc="upper left")
ax.set_title("Derived KV capacity scaling")
ax.text(0.01, -0.31, "Formula: 2 L Hkv dh bkv S; L=80 and dh=128. MHA/GQA labels specify KV heads.",
        transform=ax.transAxes, fontsize=6.2, va="top")
save(fig, "fig3_kv_scaling.pdf")


# Figure 4: paged KV author redraw + exact vLLM memory-saving data
vllm_rows=[]
for n,v in zip([2,4,6],[6.09,8.53,9.79]): vllm_rows.append({"mode":"parallel sampling","width_or_sequences":n,"memory_saving_percent":v})
for n,v in zip([2,4,6],[37.56,53.13,55.16]): vllm_rows.append({"mode":"beam search","width_or_sequences":n,"memory_saving_percent":v})
write_csv("fig4_vllm_memory_saving_fig15.csv", vllm_rows)
fig = plt.figure(figsize=(7.15, 2.55))
gs = fig.add_gridspec(1,2,width_ratios=[1.28,1.0],wspace=0.28)
ax=fig.add_subplot(gs[0,0]); ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis("off")
ax.set_title("(a) Logical tables, one shared physical prefix", pad=3)
# Two logical tables share the same prefix page IDs
for label,y,ids in [("Sequence A",0.79,[0,1,4,7]),("Sequence B",0.62,[0,1,4,9])]:
    ax.text(0.04,y+0.042,label,fontsize=7.4,va="center")
    for j,bid in enumerate(ids):
        shared=j<3
        r=Rectangle((0.24+j*0.105,y),0.075,0.085,ec="black",fc="0.82" if shared else "white",hatch="//" if shared else "",lw=0.75)
        ax.add_patch(r); ax.text(0.277+j*0.105,y+0.043,str(bid),ha="center",va="center",fontsize=7)
ax.text(0.34,0.54,"shared prefix IDs",fontsize=6.7,ha="center")
ax.text(0.625,0.54,"private suffix",fontsize=6.7,ha="center")
# Physical allocation shown by ownership group rather than crossing arrows
ax.text(0.04,0.39,"Physical pages",fontsize=7.5,fontweight="bold")
ax.text(0.04,0.27,"Shared prefix",fontsize=7.0,va="center")
for j,bid in enumerate([0,1,4]):
    r=Rectangle((0.24+j*0.105,0.225),0.075,0.085,ec="black",fc="0.82",hatch="//",lw=0.75)
    ax.add_patch(r); ax.text(0.277+j*0.105,0.268,str(bid),ha="center",va="center",fontsize=7)
ax.text(0.04,0.12,"A suffix",fontsize=7.0,va="center")
r=Rectangle((0.24,0.077),0.075,0.085,ec="black",fc="white",lw=0.75); ax.add_patch(r); ax.text(0.277,0.12,"7",ha="center",va="center",fontsize=7)
ax.text(0.39,0.12,"B suffix",fontsize=7.0,va="center")
r=Rectangle((0.51,0.077),0.075,0.085,ec="black",fc="white",lw=0.75); ax.add_patch(r); ax.text(0.547,0.12,"9",ha="center",va="center",fontsize=7)

ax2=fig.add_subplot(gs[0,1])
widths=np.array([2,4,6]); x=np.arange(3); w=0.34
pvals=[6.09,8.53,9.79]; beam=[37.56,53.13,55.16]
b1=ax2.bar(x-w/2,pvals,w,label="Parallel sampling",edgecolor="black",facecolor="0.86",hatch="//")
b2=ax2.bar(x+w/2,beam,w,label="Beam search",edgecolor="black",facecolor="white",hatch="xx")
ax2.set_xticks(x,["2","4","6"]); ax2.set_xlabel("Output sequences / beam width")
ax2.set_ylabel("KV memory saving (%)"); ax2.set_ylim(0,64); ax2.grid(axis="y",alpha=0.3)
ax2.set_title("(b) vLLM reported KV-sharing savings")
ax2.legend(frameon=False,loc="upper left")
for bars in [b1,b2]:
    for b in bars: ax2.text(b.get_x()+b.get_width()/2,b.get_height()+1.3,f"{b.get_height():.2f}",ha="center",fontsize=6.4)
ax2.text(0.01,-0.29,"Exact values from Kwon et al., Fig. 15 (OPT-13B, Alpaca trace).",transform=ax2.transAxes,fontsize=6.1,va="top")
save(fig,"fig4_paged_kv.pdf")


# Figure 5: continuous batching redraw + exact Sarathi capacity maxima
sarathi_rows=[
    {"model_hardware":"Mistral-7B / 1x A100","capacity_gain_x":2.6},
    {"model_hardware":"Yi-34B / 2x A100","capacity_gain_x":3.7},
    {"model_hardware":"Falcon-180B / 8x A100 PP+TP","capacity_gain_x":5.6},
]
write_csv("fig5_sarathi_capacity_gains.csv",sarathi_rows)
fig=plt.figure(figsize=(7.15,2.55)); gs=fig.add_gridspec(1,2,width_ratios=[1.45,1],wspace=0.30)
ax=fig.add_subplot(gs[0,0]); ax.set_xlim(-1.8,12.4); ax.set_ylim(-0.58,4.35); ax.axis("off")
ax.set_title("(a) Iteration-level admission with bounded prefill chunks",pad=3)
for y,name in zip([3.35,2.35,1.35,0.35],["R1","R2","R3","R4"]):
    ax.text(-0.45,y,name,ha="right",va="center",fontsize=7.2)
    ax.hlines(y,0,12,color="0.82",lw=0.5)
segments={
    3.35:[(0,1.0,"P1","//"),(1.0,5.4,"decode",""),(6.4,1.6,"decode","" )],
    2.35:[(1.2,1.0,"P2a","//"),(2.2,1.0,"P2b","//"),(3.2,4.8,"decode","")],
    1.35:[(4.4,1.0,"P3","//"),(5.4,4.0,"decode","")],
    0.35:[(8.2,1.0,"P4","//"),(9.2,2.4,"decode","")],
}
for y,segs in segments.items():
    for st,ln,lab,hatch in segs:
        rect=Rectangle((st,y-0.25),ln,0.50,ec="black",fc="white" if hatch else "0.86",hatch=hatch,lw=0.7)
        ax.add_patch(rect); ax.text(st+ln/2,y,lab,ha="center",va="center",fontsize=6.2)
for t in range(13): ax.text(t,-0.04,str(t),ha="center",va="top",fontsize=6)
ax.text(6,-0.47,"Scheduler iteration",ha="center",fontsize=6.5)
ax.text(0,3.88,"Completed requests retire; new work enters at iteration boundaries.",fontsize=6.2)

ax2=fig.add_subplot(gs[0,1])
labels=["Mistral-7B\n1 A100","Yi-34B\n2 A100","Falcon-180B\n8 A100 PP+TP"]
vals=[r["capacity_gain_x"] for r in sarathi_rows]
y=np.arange(len(vals))
bars=ax2.barh(y,vals,edgecolor="black",facecolor="0.84",hatch=["//","xx",".."])
ax2.set_yticks(y,labels); ax2.invert_yaxis(); ax2.set_xlim(0,6.2); ax2.set_xlabel("Maximum serving-capacity gain vs vLLM (x)")
ax2.grid(axis="x",alpha=0.3); ax2.set_title("(b) Sarathi-Serve reported maxima")
for b,v in zip(bars,vals): ax2.text(v+0.10,b.get_y()+b.get_height()/2,f"{v:.1f}x",va="center",fontsize=7)
ax2.text(0.01,-0.24,"Published OSDI'24 results; configurations and SLO targets differ.",transform=ax2.transAxes,fontsize=6.1,va="top")
save(fig,"fig5_continuous_batching.pdf")


# Figure 6: speculative diagram + exact Medusa speedups
medusa_rows=[
    {"model":"Vicuna-7B","baseline":1.0,"Medusa-1":2.18,"Medusa-2":2.83},
    {"model":"Vicuna-13B","baseline":1.0,"Medusa-1":2.33,"Medusa-2":2.83},
]
write_csv("fig6_medusa_speedups.csv",medusa_rows)
fig=plt.figure(figsize=(7.15,2.45)); gs=fig.add_gridspec(1,2,width_ratios=[1.25,1],wspace=0.30)
ax=fig.add_subplot(gs[0,0]); ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis("off")
ax.set_title("(a) Draft-and-verify dataflow",pad=3)
box(ax,(0.04,0.54),0.24,0.22,"Draft proposes\n$k$ candidates",fc="white",fontsize=7.4)
box(ax,(0.38,0.54),0.24,0.22,"Target verifies\nall $k$ positions once",fc="0.86",fontsize=7.4)
box(ax,(0.72,0.54),0.24,0.22,"Accept prefix\nand correct rejection",fc="white",fontsize=7.2)
arrow(ax,(0.28,0.65),(0.38,0.65)); arrow(ax,(0.62,0.65),(0.72,0.65))
box(ax,(0.38,0.18),0.24,0.17,"Commit accepted\nKV state",fc="0.92",fontsize=7.2)
arrow(ax,(0.84,0.54),(0.62,0.35)); arrow(ax,(0.38,0.265),(0.16,0.54),connectionstyle="arc3,rad=-0.22")

ax2=fig.add_subplot(gs[0,1])
models=[r["model"] for r in medusa_rows]; x=np.arange(2); w=0.24
series=[("Autoregressive",[1,1],""),("Medusa-1",[2.18,2.33],"//"),("Medusa-2",[2.83,2.83],"xx")]
for i,(name,vals,hatch) in enumerate(series):
    bars=ax2.bar(x+(i-1)*w,vals,w,label=name,edgecolor="black",facecolor=str(0.92-0.16*i),hatch=hatch)
    for b,v in zip(bars,vals): ax2.text(b.get_x()+b.get_width()/2,v+0.07,f"{v:.2f}",ha="center",fontsize=6.3)
ax2.set_xticks(x,models); ax2.set_ylim(0,3.25); ax2.set_ylabel("Wall-time speedup (x)"); ax2.grid(axis="y",alpha=0.3)
ax2.set_title("(b) Medusa reported speedups")
ax2.legend(frameon=False,ncol=3,loc="upper center",bbox_to_anchor=(0.5,-0.16))
ax2.text(0.01,-0.35,"Exact values for Vicuna 7B/13B; baseline is optimized autoregressive decoding.",transform=ax2.transAxes,fontsize=6.1,va="top")
save(fig,"fig6_speculative.pdf")


# Figure 7: exact published measurements for KIVI and DistServe (not common metric)
kivi_rows=[
    {"metric":"peak-memory reduction factor","low":2.6,"high":2.6},
    {"metric":"maximum batch-size factor","low":4.0,"high":4.0},
    {"metric":"throughput speedup range","low":2.35,"high":3.47},
]
dist_rows=[
    {"baseline":"vLLM","request_rate_gain_x":4.3,"tighter_SLO_x":12.6},
    {"baseline":"DeepSpeed-MII","request_rate_gain_x":1.8,"tighter_SLO_x":2.6},
]
write_csv("fig7_kivi_reported_metrics.csv",kivi_rows); write_csv("fig7_distserve_summarization.csv",dist_rows)
fig=plt.figure(figsize=(7.15,2.45)); gs=fig.add_gridspec(1,2,wspace=0.28)
ax=fig.add_subplot(gs[0,0])
labels=["Peak memory\nreduction","Max batch\nsize","Throughput\nspeedup"]
y=np.arange(3)
for i,r in enumerate(kivi_rows):
    low,high=r["low"],r["high"]
    ax.hlines(i,low,high,lw=5,color="0.45")
    ax.plot([low,high],[i,i],"o",ms=4,color="black")
    text=f"{low:.2f}x" if low==high else f"{low:.2f}-{high:.2f}x"
    ax.text(high+0.10,i,text,va="center",fontsize=7)
ax.set_yticks(y,labels); ax.invert_yaxis(); ax.set_xlim(0,4.6); ax.set_xlabel("Reported factor (x)"); ax.grid(axis="x",alpha=0.3)
ax.set_title("(a) KIVI 2-bit KV-cache results")
ax.text(0.01,-0.25,"Published across Llama, Falcon, and Mistral workloads; metrics are distinct.",transform=ax.transAxes,fontsize=6.1,va="top")

ax2=fig.add_subplot(gs[0,1])
base=[r["baseline"] for r in dist_rows]; x=np.arange(2); w=0.34
b1=ax2.bar(x-w/2,[r["request_rate_gain_x"] for r in dist_rows],w,label="Higher request rate",edgecolor="black",facecolor="0.83",hatch="//")
b2=ax2.bar(x+w/2,[r["tighter_SLO_x"] for r in dist_rows],w,label="Tighter SLO",edgecolor="black",facecolor="white",hatch="xx")
ax2.set_xticks(x,base); ax2.set_ylim(0,14.2); ax2.set_ylabel("Improvement factor (x)"); ax2.grid(axis="y",alpha=0.3)
ax2.set_title("(b) DistServe, OPT-66B summarization")
ax2.legend(frameon=False,loc="upper left")
for bars in [b1,b2]:
    for b in bars: ax2.text(b.get_x()+b.get_width()/2,b.get_height()+0.3,f"{b.get_height():.1f}x",ha="center",fontsize=6.5)
ax2.text(0.01,-0.25,"Exact comparisons reported in OSDI'24; >90% of requests satisfy latency constraints.",transform=ax2.transAxes,fontsize=6.1,va="top")
save(fig,"fig7_reported_evidence.pdf")


# Figure 8: prefill/decode disaggregation -- author redraw of published architecture
fig=plt.figure(figsize=(7.15,2.30))
ax=fig.add_subplot(111); ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis("off")
ax.set_title("Prefill/decode disaggregation and phase-specific control", pad=3)
items=[
    (0.02,0.55,0.13,0.22,"Request\nqueue","0.94"),
    (0.19,0.55,0.17,0.22,"Prefill\nscheduler","white"),
    (0.40,0.55,0.18,0.22,"Prefill pool\ncompute-oriented","0.86"),
    (0.40,0.17,0.18,0.22,"Decode pool\nbandwidth-oriented","0.86"),
    (0.63,0.17,0.17,0.22,"Decode\nscheduler","white"),
    (0.84,0.36,0.13,0.22,"Token\nstream","0.94"),
]
for x,y,w,h,t,fc in items: box(ax,(x,y),w,h,t,fc=fc,fontsize=7.5)
arrow(ax,(0.15,0.66),(0.19,0.66)); arrow(ax,(0.36,0.66),(0.40,0.66)); arrow(ax,(0.58,0.66),(0.84,0.52))
arrow(ax,(0.49,0.55),(0.49,0.39)); arrow(ax,(0.58,0.28),(0.63,0.28)); arrow(ax,(0.80,0.28),(0.84,0.42))
ax.text(0.165,0.70,"prompt",fontsize=6.2,ha="center")
ax.text(0.38,0.70,"admit",fontsize=6.2,ha="center")
ax.text(0.70,0.68,"first token",fontsize=6.2,ha="center")
ax.text(0.53,0.46,"layer-wise KV transfer",fontsize=6.2,ha="left")
ax.text(0.605,0.32,"ready",fontsize=6.2,ha="center")
ax.text(0.82,0.31,"decode tokens",fontsize=6.2,ha="center")
ax.text(0.19,0.47,"TTFT policy",fontsize=6.3,ha="center")
ax.text(0.71,0.09,"TPOT/goodput policy",fontsize=6.3,ha="center")
save(fig,"fig8_pd_disaggregation.pdf")


# Figure 10: authentic MoE architecture redraw + exact MegaScale-Infer evidence
mega_rows=[
    {"comparison":"Scaled-MoE vs vLLM","per_GPU_decode_throughput_gain_x":7.11},
    {"comparison":"Scaled-MoE vs TensorRT-LLM","per_GPU_decode_throughput_gain_x":1.90},
    {"comparison":"M2N library vs NCCL at 256KB","throughput_gain_x":4.2,"median_latency_reduction_percent":68.2},
]
write_csv("fig10_megascale_infer_metrics.csv",mega_rows)
fig=plt.figure(figsize=(7.15,2.55)); gs=fig.add_gridspec(1,2,width_ratios=[1.38,1],wspace=0.30)
ax=fig.add_subplot(gs[0,0]); ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis("off")
ax.set_title("(a) Disaggregated attention/expert serving",pad=3)
box(ax,(0.03,0.48),0.20,0.22,"Micro-batch\n+ KV state",fc="0.92",fontsize=7.4)
box(ax,(0.36,0.48),0.22,0.22,"Attention pool\nKV + attention",fc="white",fontsize=7.4)
box(ax,(0.72,0.48),0.22,0.22,"Expert/FFN pool\nexpert parallelism",fc="0.86",fontsize=7.2)
arrow(ax,(0.23,0.59),(0.36,0.59)); arrow(ax,(0.58,0.59),(0.72,0.59))
arrow(ax,(0.83,0.48),(0.47,0.31),connectionstyle="arc3,rad=-0.18")

ax2=fig.add_subplot(gs[0,1])
labels=["Per-GPU decode\nvs vLLM","Per-GPU decode\nvs TensorRT-LLM","M2N throughput\nvs NCCL"]
vals=[7.11,1.90,4.2]
y=np.arange(3)
bars=ax2.barh(y,vals,edgecolor="black",facecolor="0.84",hatch=["//","xx",".."])
ax2.set_yticks(y,labels); ax2.invert_yaxis(); ax2.set_xlim(0,7.8); ax2.set_xlabel("Reported gain (x)"); ax2.grid(axis="x",alpha=0.3)
ax2.set_title("(b) MegaScale-Infer measurements")
for b,v in zip(bars,vals): ax2.text(v+0.12,b.get_y()+b.get_height()/2,f"{v:.2f}x",va="center",fontsize=7)
ax2.text(0.01,-0.23,"Scaled-MoE comparisons; M2N result is at 256-KB messages and also reports 68.2% lower median latency.",transform=ax2.transAxes,fontsize=6.1,va="top")
save(fig,"fig10_moe_serving.pdf")


# Figure 9: RL dataflow redraw + exact OpenRLHF 14B step-time data
openrl_rows=[]
contexts=[1,2,4,8]; ours=[25.5,51.0,136.3,328.6]; verl=[28.5,74.3,202.8,511.1]
for c,o,v in zip(contexts,ours,verl): openrl_rows.append({"model":"14B","max_generation_K_tokens":c,"OpenRLHF_seconds_per_step":o,"verl_seconds_per_step":v,"speedup_x":v/o})
write_csv("fig9_openrlhf_14b_step_times.csv",openrl_rows)
fig=plt.figure(figsize=(7.15,2.55)); gs=fig.add_gridspec(1,2,width_ratios=[1.40,1],wspace=0.30)
ax=fig.add_subplot(gs[0,0]); ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis("off")
ax.set_title("(a) Inference-centered RL post-training dataflow",pad=3)
# Clockwise six-stage loop with short labels
coords=[
    (0.02,0.62,0.17,0.18,"Prompts / tasks","0.92"),
    (0.30,0.62,0.19,0.18,"Rollout engine\n(inference)","white"),
    (0.60,0.62,0.18,0.18,"Environment /\ntools","0.92"),
    (0.60,0.18,0.18,0.18,"Reward /\nverification","white"),
    (0.30,0.18,0.19,0.18,"Policy update","0.86"),
    (0.02,0.18,0.17,0.18,"Weight refresh\n/ resharding","white"),
]
for x,y,w,h,t,fc in coords: box(ax,(x,y),w,h,t,fc=fc,fontsize=7.1)
arrow(ax,(0.19,0.71),(0.30,0.71)); arrow(ax,(0.49,0.71),(0.60,0.71)); arrow(ax,(0.69,0.62),(0.69,0.36)); arrow(ax,(0.60,0.27),(0.49,0.27)); arrow(ax,(0.30,0.27),(0.19,0.27)); arrow(ax,(0.105,0.36),(0.105,0.62))

ax2=fig.add_subplot(gs[0,1])
x=np.arange(4); w=0.34
b1=ax2.bar(x-w/2,ours,w,label="OpenRLHF",edgecolor="black",facecolor="0.82",hatch="//")
b2=ax2.bar(x+w/2,verl,w,label="verl",edgecolor="black",facecolor="white",hatch="xx")
ax2.set_xticks(x,["1K","2K","4K","8K"]); ax2.set_xlabel("Maximum generation length")
ax2.set_ylabel("Average training-step time (s)"); ax2.grid(axis="y",alpha=0.3); ax2.set_title("(b) Published 14B long-CoT RLVR times")
ax2.legend(frameon=False,loc="upper left")
for b in list(b1)+list(b2): ax2.text(b.get_x()+b.get_width()/2,b.get_height()+7,f"{b.get_height():.1f}",ha="center",fontsize=5.9,rotation=90,va="bottom")
ax2.text(0.01,-0.25,"Exact Table 1 values: 8 H200 GPUs, identical model/algorithm settings, first 10 steps excluded.",transform=ax2.transAxes,fontsize=6.1,va="top")
save(fig,"fig9_rl_loop.pdf")

print(f"Wrote figures to {FIG} and source data to {DATA}")
