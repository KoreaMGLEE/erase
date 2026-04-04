"""Fig 3 & Fig 4 final: publication style matching fig1/fig2 tone."""
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter
from datasets import load_dataset

FIG_DIR = "/workspace/erase/figures"
OUTPUT_DIR = "/workspace/erase/outputs/plan5/analysis"
os.makedirs(FIG_DIR, exist_ok=True)

# ── Shared style ─────────────────────────────────────────────────

def set_pub_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "dejavuserif",
        "axes.linewidth": 0.4,
        "xtick.major.width": 0.4,
        "ytick.major.width": 0.4,
        "xtick.minor.width": 0.3,
        "ytick.minor.width": 0.3,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "xtick.minor.size": 1.5,
        "ytick.minor.size": 1.5,
        "xtick.direction": "in",
        "ytick.direction": "in",
    })

FAMILY_COLORS = {"BERT": "#1f77b4", "T5 v1.1": "#d62728", "Pythia": "#2ca02c"}
FAMILY_MARKERS = {"BERT": "o", "T5 v1.1": "s", "Pythia": "^"}

# ── Model config ─────────────────────────────────────────────────

MODELS = {
    "bert-mini": {"params": 11e6, "family": "BERT"},
    "bert-small": {"params": 29e6, "family": "BERT"},
    "bert-medium": {"params": 42e6, "family": "BERT"},
    "bert-base": {"params": 110e6, "family": "BERT"},
    "bert-large": {"params": 335e6, "family": "BERT"},
    "t5-v1_1-small": {"params": 77e6, "family": "T5 v1.1"},
    "t5-v1_1-base": {"params": 248e6, "family": "T5 v1.1"},
    "t5-v1_1-large": {"params": 783e6, "family": "T5 v1.1"},
    "t5-v1_1-xl": {"params": 3e9, "family": "T5 v1.1"},
    "t5-v1_1-xxl": {"params": 11e9, "family": "T5 v1.1"},
    "pythia-14m": {"params": 14e6, "family": "Pythia"},
    "pythia-31m": {"params": 31e6, "family": "Pythia"},
    "pythia-70m": {"params": 70e6, "family": "Pythia"},
    "pythia-160m": {"params": 160e6, "family": "Pythia"},
    "pythia-410m": {"params": 410e6, "family": "Pythia"},
    "pythia-1b": {"params": 1e9, "family": "Pythia"},
    "pythia-1.4b": {"params": 1.4e9, "family": "Pythia"},
    "pythia-2.8b": {"params": 2.8e9, "family": "Pythia"},
    "pythia-6.9b": {"params": 6.9e9, "family": "Pythia"},
    "pythia-12b": {"params": 12e9, "family": "Pythia"},
}

NEGATION_WORDS = {"no", "not", "never", "nothing", "nobody", "neither",
                  "nor", "none", "cannot", "can't", "don't", "doesn't",
                  "didn't", "won't", "wouldn't", "shouldn't", "couldn't",
                  "isn't", "aren't", "wasn't", "weren't"}
K = 30


def load_split_confidences(model_name):
    split_confs = {}
    for split in ["split1", "split2", "split3"]:
        candidates = [
            f"/workspace/erase/outputs/plan0/confidence/{model_name}_{split}/avg_conf.json",
            f"/workspace/erase/outputs/plan5/confidence/{model_name}_{split}/avg_conf.json",
            f"/workspace/erase/outputs/plan5/confidence/{model_name}_sentinel_{split}/avg_conf.json",
            f"/workspace/erase/outputs/plan5/confidence/{model_name}_sentinel_1e-4_{split}/avg_conf.json",
        ]
        for path in candidates:
            if os.path.exists(path):
                with open(path) as f:
                    split_confs[split] = json.load(f)
                break
    return split_confs


def get_easy_set(conf_dict, k_pct):
    items = sorted(conf_dict.items(), key=lambda x: -x[1])
    n = int(len(items) * k_pct / 100)
    return set(item[0] for item in items[:n])


def has_negation(hypothesis):
    tokens = set(hypothesis.lower().split())
    return len(tokens & NEGATION_WORDS) > 0


def lexical_overlap(premise, hypothesis):
    p_tokens = set(premise.lower().split())
    h_tokens = set(hypothesis.lower().split())
    if len(h_tokens) == 0:
        return 0.0
    return len(p_tokens & h_tokens) / len(h_tokens)


# ══════════════════════════════════════════════════════════════════
# FIG 3: Scale vs Shortcut
# ══════════════════════════════════════════════════════════════════

def compute_fig3_data():
    """Compute or load cached shortcut data."""
    cache_path = os.path.join(OUTPUT_DIR, "shortcut_by_model_split_avg.json")
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            return json.load(f)

    print("  Computing from scratch (loading MNLI)...")
    ds = load_dataset("glue", "mnli", split="train")
    with open("/workspace/erase/outputs/plan0/data/all_90k_indices.json") as f:
        all_90k = json.load(f)

    idx_to_data = {}
    for idx in all_90k:
        item = ds[idx]
        idx_to_data[str(idx)] = {
            "premise": item["premise"],
            "hypothesis": item["hypothesis"],
            "label": item["label"],
        }

    all_contradiction = [idx for idx, d in idx_to_data.items() if d["label"] == 2]
    all_entailment = [idx for idx, d in idx_to_data.items() if d["label"] == 0]
    global_negation = sum(has_negation(idx_to_data[idx]["hypothesis"]) for idx in all_contradiction) / len(all_contradiction)
    global_overlap = sum(lexical_overlap(idx_to_data[idx]["premise"], idx_to_data[idx]["hypothesis"]) >= 0.6
                         for idx in all_entailment) / len(all_entailment)

    results = {"global_negation": global_negation, "global_overlap": global_overlap, "models": {}}
    for model_name, info in MODELS.items():
        split_confs = load_split_confidences(model_name)
        if not split_confs:
            continue
        neg_ratios, ovlp_ratios = [], []
        for split, conf in split_confs.items():
            easy_set = get_easy_set(conf, K)
            easy_contr = [idx for idx in easy_set if idx_to_data.get(idx, {}).get("label") == 2]
            easy_entail = [idx for idx in easy_set if idx_to_data.get(idx, {}).get("label") == 0]
            if easy_contr:
                neg_ratios.append(sum(has_negation(idx_to_data[idx]["hypothesis"]) for idx in easy_contr) / len(easy_contr))
            if easy_entail:
                ovlp_ratios.append(sum(lexical_overlap(idx_to_data[idx]["premise"], idx_to_data[idx]["hypothesis"]) >= 0.6
                                       for idx in easy_entail) / len(easy_entail))
        results["models"][model_name] = {
            "params": info["params"], "family": info["family"],
            "neg_mean": float(np.mean(neg_ratios)) if neg_ratios else 0,
            "neg_std": float(np.std(neg_ratios)) if neg_ratios else 0,
            "ovlp_mean": float(np.mean(ovlp_ratios)) if ovlp_ratios else 0,
            "ovlp_std": float(np.std(ovlp_ratios)) if ovlp_ratios else 0,
        }
        print(f"    {model_name}: neg={results['models'][model_name]['neg_mean']:.3f}, ovlp={results['models'][model_name]['ovlp_mean']:.3f}")

    with open(cache_path, "w") as f:
        json.dump(results, f, indent=2)
    return results


def draw_fig3_fig4_combined(results):
    """Combined 3-panel figure: (a) negation, (b) overlap, (c) knowledge lollipop."""
    set_pub_style()
    import matplotlib.gridspec as gridspec

    # ── Shared font sizes ──
    title_fs  = 7.0
    label_fs  = 6.5
    tick_fs   = 5.5
    legend_fs = 5.5
    annot_fs  = 5.0
    detail_fs = 4.2

    lw = 0.8
    ms = 2.5
    shade_alpha = 0.04

    # Match fig2 aspect (~8.5 x 2.8); tighter wspace, wider (c)
    fig = plt.figure(figsize=(8.5, 2.5))

    # (a),(b) tighter + (c) wider & shorter
    gs_outer = gridspec.GridSpec(1, 3, width_ratios=[0.95, 0.95, 0.95],
                                 wspace=0.22, left=0.065, right=0.97,
                                 bottom=0.18, top=0.88)
    ax1 = fig.add_subplot(gs_outer[0])
    ax2 = fig.add_subplot(gs_outer[1])

    # Panel (c): 85% height, vertically centred
    gs_inner = gridspec.GridSpecFromSubplotSpec(
        3, 1, subplot_spec=gs_outer[2],
        height_ratios=[0.075, 0.85, 0.075])
    ax3 = fig.add_subplot(gs_inner[1])

    # ═══ Panels (a) & (b): line plots ═══
    line_panels = [
        (ax1, "neg", "Negation ratio",
         r"$\bf{(a)}$" + " Scale-sensitive shortcut",
         results["global_negation"]),
        (ax2, "ovlp", "Lexical overlap ratio",
         r"$\bf{(b)}$" + " Architecture-sensitive shortcut",
         results["global_overlap"]),
    ]

    for ax, metric, ylabel, title, global_val in line_panels:
        for family in ["BERT", "T5 v1.1", "Pythia"]:
            xs, ys, errs = [], [], []
            for model_name, data in results["models"].items():
                if data["family"] == family:
                    xs.append(data["params"])
                    ys.append(data[f"{metric}_mean"])
                    errs.append(data[f"{metric}_std"])
            if not xs:
                continue
            order = np.argsort(xs)
            xs  = np.array([xs[i]   for i in order])
            ys  = np.array([ys[i]   for i in order])
            errs = np.array([errs[i] for i in order])

            ax.plot(xs, ys, color=FAMILY_COLORS[family], marker=FAMILY_MARKERS[family],
                    label=family, linewidth=lw, markersize=ms,
                    markeredgecolor=FAMILY_COLORS[family], markeredgewidth=0.3,
                    markerfacecolor=FAMILY_COLORS[family], zorder=3)
            ax.fill_between(xs, ys - errs, ys + errs,
                            color=FAMILY_COLORS[family], alpha=shade_alpha, zorder=1)

        ax.axhline(y=global_val, color="#888888", linestyle="--", linewidth=0.5, alpha=0.6, zorder=0)
        # Baseline label — right side of each panel, near the dashed line
        ax.text(0.97, 0.06, "global baseline", fontsize=tick_fs - 1, color="#777777",
                ha="right", transform=ax.transAxes, style="italic")

        ax.set_xscale("log")
        ax.set_xlabel("Parameters", fontsize=label_fs, labelpad=2)
        ax.set_ylabel(ylabel, fontsize=label_fs, labelpad=2)
        ax.set_title(title, fontsize=title_fs, pad=4)
        ax.tick_params(labelsize=tick_fs, pad=2)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.xaxis.set_major_locator(LogLocator(base=10, numticks=5))
        ax.xaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10), numticks=20))
        ax.xaxis.set_minor_formatter(NullFormatter())

    # Tighten right panel y-axis
    ovlp_vals = [d["ovlp_mean"] for d in results["models"].values() if d["ovlp_mean"] > 0]
    if ovlp_vals:
        lo = min(min(ovlp_vals), results["global_overlap"]) - 0.03
        hi = max(ovlp_vals) + 0.03
        ax2.set_ylim(lo, hi)

    # Legend inside panel (b) lower-left — away from baseline label (right)
    handles, labels = ax1.get_legend_handles_labels()
    ax2.legend(handles, labels, fontsize=legend_fs, loc="lower left",
               frameon=True, framealpha=0.9, edgecolor="none",
               handlelength=1.2, handletextpad=0.3,
               borderpad=0.3, labelspacing=0.3)

    # ═══ Panel (c): lollipop ═══
    BASE  = {"S": 129, "F": 316, "P": 19}
    HE_MH = {"S": 13,  "F": 20,  "P": 7}
    n_base = sum(BASE.values())
    n_hemh = sum(HE_MH.values())
    base_pct = {k: v / n_base * 100 for k, v in BASE.items()}
    hemh_pct = {k: v / n_hemh * 100 for k, v in HE_MH.items()}
    delta = {k: hemh_pct[k] - base_pct[k] for k in BASE}

    keys    = ["F",       "S",           "P"]
    ylabels = ["Factual", "Situational", "Procedural"]

    color_over  = "#C24E24"
    color_under = "#345B73"     # deep steel blue

    y = np.arange(len(ylabels))
    deltas = [delta[k] for k in keys]
    colors = [color_over if d > 0 else color_under for d in deltas]

    for i, (d, c) in enumerate(zip(deltas, colors)):
        ax3.plot([0, d], [y[i], y[i]], color=c, linewidth=0.8,
                 solid_capstyle="round", zorder=2)
        ax3.plot(d, y[i], "o", color=c, markersize=4.5, zorder=3,
                 markeredgecolor="white", markeredgewidth=0.3)

    # Annotations: pp above stem, detail below — uniform layout
    pp_dy     = -0.04          # pp label slightly above stem
    detail_dy = +0.16          # detail just below — tight & uniform
    for i, k in enumerate(keys):
        d = deltas[i]
        actual = hemh_pct[k]
        base = base_pct[k]
        # pp label: outside dot
        pp_offset = 1.0 if d >= 0 else -1.0
        pp_ha = "left" if d >= 0 else "right"
        ax3.text(d + pp_offset, y[i] + pp_dy, f"{d:+.0f}pp",
                 ha=pp_ha, va="center", fontsize=annot_fs,
                 color=colors[i], fontweight="medium")
        # detail: same ha as pp, pulled inward for negatives to avoid overflow
        det_offset = pp_offset if d >= 0 else pp_offset + 3.0
        ax3.text(d + det_offset, y[i] + detail_dy, f"({actual:.0f}% vs {base:.0f}%)",
                 ha=pp_ha, va="center", fontsize=detail_fs, color="#777777")

    ax3.axvline(x=0, color="#444444", linewidth=0.4, zorder=1)
    ax3.set_yticks(y)
    ax3.set_yticklabels(ylabels, fontsize=tick_fs)
    ax3.set_xlabel("Diff. from base rate (pp)", fontsize=label_fs, labelpad=2)
    ax3.set_xlim(-26, 22)
    ax3.set_ylim(y[-1] + 0.50, y[0] - 0.50)
    ax3.tick_params(axis="x", labelsize=tick_fs, pad=2)
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)
    ax3.set_title(r"$\bf{(c)}$" + " Knowledge type shift",
                  fontsize=title_fs, pad=4)

    fig.savefig(os.path.join(FIG_DIR, "fig3_fig4_combined.pdf"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(FIG_DIR, "fig3_fig4_combined.png"), dpi=300, bbox_inches="tight")
    print("  Saved: fig3_fig4_combined")
    plt.close(fig)


if __name__ == "__main__":
    print("Fig 3+4 combined:")
    results = compute_fig3_data()
    draw_fig3_fig4_combined(results)

    print("Done!")
