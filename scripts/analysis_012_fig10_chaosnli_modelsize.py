"""Fig 10: ChaosNLI — Human agreement vs model confidence by model size."""
import json, os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

OUT_DIR = "/workspace/erase/outputs/plan12_analysis"

def main():
    with open(os.path.join(OUT_DIR, "chaosnli_model_confidence.json")) as f:
        data = json.load(f)

    models = ["bert-mini", "bert-small", "bert-base"]
    params = {"bert-mini": "11M", "bert-small": "29M", "bert-base": "110M"}
    colors = {"bert-mini": "#e74c3c", "bert-small": "#e67e22", "bert-base": "#2980b9"}

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("ChaosNLI: Human Agreement vs Model Confidence by Model Size", fontsize=14, fontweight="bold")

    # --- Panel 1: Scatter — human agreement vs model confidence ---
    ax = axes[0, 0]
    for model in models:
        per_ex = data[model]["per_example"]
        agreements = [v["agreement"] for v in per_ex.values()]
        confs = [v["model_conf"] for v in per_ex.values()]
        r = np.corrcoef(agreements, confs)[0, 1]
        ax.scatter(agreements, confs, alpha=0.08, s=8, color=colors[model], label=f"{model} ({params[model]}) r={r:.3f}")
        # Trend line
        z = np.polyfit(agreements, confs, 1)
        x_line = np.linspace(0.2, 1.0, 50)
        ax.plot(x_line, np.polyval(z, x_line), color=colors[model], linewidth=2, linestyle="--")
    ax.set_xlabel("Human Agreement (ChaosNLI)")
    ax.set_ylabel("Model Confidence (on gold label)")
    ax.set_title("Scatter: Human Agreement vs Model Confidence")
    ax.legend(fontsize=9, loc="upper left")
    ax.set_xlim(0.15, 1.05)

    # --- Panel 2: Binned — human agreement bins → model confidence ---
    ax = axes[0, 1]
    bins = [(0, 0.4, "<40%"), (0.4, 0.5, "40-50%"), (0.5, 0.6, "50-60%"),
            (0.6, 0.7, "60-70%"), (0.7, 0.8, "70-80%"), (0.8, 0.9, "80-90%"), (0.9, 1.01, "90%+")]
    bin_centers = np.arange(len(bins))

    for model in models:
        per_ex = data[model]["per_example"]
        means = []
        for lo, hi, label in bins:
            subset = [v["model_conf"] for v in per_ex.values() if lo <= v["agreement"] < hi]
            means.append(np.mean(subset) if subset else 0)
        ax.plot(bin_centers, means, marker="o", color=colors[model], linewidth=2,
                label=f"{model} ({params[model]})", markersize=6)

    ax.set_xticks(bin_centers)
    ax.set_xticklabels([b[2] for b in bins], rotation=30, ha="right", fontsize=8)
    ax.set_xlabel("Human Agreement Bin")
    ax.set_ylabel("Mean Model Confidence")
    ax.set_title("Model Confidence by Human Agreement Bin")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # --- Panel 3: Binned — human agreement bins → model accuracy ---
    ax = axes[1, 0]
    for model in models:
        per_ex = data[model]["per_example"]
        accs = []
        for lo, hi, label in bins:
            subset = [v["correct"] for v in per_ex.values() if lo <= v["agreement"] < hi]
            accs.append(np.mean(subset) if subset else 0)
        ax.plot(bin_centers, accs, marker="s", color=colors[model], linewidth=2,
                label=f"{model} ({params[model]})", markersize=6)

    ax.axhline(1/3, color="gray", linestyle=":", alpha=0.5, label="Chance (33%)")
    ax.set_xticks(bin_centers)
    ax.set_xticklabels([b[2] for b in bins], rotation=30, ha="right", fontsize=8)
    ax.set_xlabel("Human Agreement Bin")
    ax.set_ylabel("Model Accuracy")
    ax.set_title("Model Accuracy by Human Agreement Bin")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # --- Panel 4: Confidence gap (model_conf difference from overall mean) by agreement bin ---
    ax = axes[1, 1]
    for model in models:
        per_ex = data[model]["per_example"]
        overall_mean = np.mean([v["model_conf"] for v in per_ex.values()])
        gaps = []
        ns = []
        for lo, hi, label in bins:
            subset = [v["model_conf"] for v in per_ex.values() if lo <= v["agreement"] < hi]
            if subset:
                gaps.append(np.mean(subset) - overall_mean)
                ns.append(len(subset))
            else:
                gaps.append(0)
                ns.append(0)
        ax.bar(bin_centers + (models.index(model) - 1) * 0.25, gaps, width=0.25,
               color=colors[model], label=f"{model} ({params[model]})", alpha=0.8)

    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xticks(bin_centers)
    ax.set_xticklabels([b[2] for b in bins], rotation=30, ha="right", fontsize=8)
    ax.set_xlabel("Human Agreement Bin")
    ax.set_ylabel("Confidence Gap (vs overall mean)")
    ax.set_title("Confidence Gap: Larger Models Align More with Humans")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    outpath = os.path.join(OUT_DIR, "figures", "fig10_chaosnli_modelsize.png")
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    print(f"Saved: {outpath}")

    # Print summary stats
    print("\n=== Summary ===")
    for model in models:
        per_ex = data[model]["per_example"]
        agreements = [v["agreement"] for v in per_ex.values()]
        confs = [v["model_conf"] for v in per_ex.values()]
        r = np.corrcoef(agreements, confs)[0, 1]
        acc = np.mean([v["correct"] for v in per_ex.values()])

        # Confidence spread: VHigh vs Low
        low = [v["model_conf"] for v in per_ex.values() if v["agreement"] < 0.5]
        vhigh = [v["model_conf"] for v in per_ex.values() if v["agreement"] >= 0.9]
        spread = np.mean(vhigh) - np.mean(low) if vhigh else 0

        print(f"  {model} ({params[model]}): r={r:.3f}, acc={acc:.3f}, "
              f"conf_spread(VHigh-Low)={spread:.3f}")


if __name__ == "__main__":
    main()
