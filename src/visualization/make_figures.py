"""
Generate plots for the final report based on Yambda-50M results.
"""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

PROCESSED = Path("data/processed")
EXPERIMENTS = Path("experiments")
FIGURES = Path("reports/figures")
FIGURES.mkdir(parents=True, exist_ok=True)

# Set style
sns.set_theme(style="whitegrid")

def plot_exhaustive():
    """Plot: Best NDCG@10 depending on LLM feature subset size."""
    path = PROCESSED / "exhaustive_results.csv"
    if not path.exists():
        print(f"WARNING: {path} not found, skipping fig1")
        return

    df = pd.read_csv(path)
    best_per_size = df.groupby("size")["ndcg"].max().reset_index()

    plt.figure(figsize=(8, 5))
    plt.plot(best_per_size["size"], best_per_size["ndcg"], marker="o", linewidth=2)
    
    # Baseline line (approx 0.6667 from logs)
    plt.axhline(y=0.6667, color="gray", linestyle="--", label="Manual baseline (valid)")
    
    plt.title("Exhaustive Search: Best NDCG@10 by LLM Subset Size")
    plt.xlabel("Number of LLM Features")
    plt.ylabel("Best NDCG@10 (Valid)")
    plt.xticks(best_per_size["size"])
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES / "fig1_exhaustive.png", dpi=150)
    plt.close()
    print("  INFO: fig1_exhaustive.png saved")

def plot_feature_importance():
    """Plot: Top-15 important features from final model."""
    path = PROCESSED / "gbdt_feature_importance.csv"
    if not path.exists():
        print(f"WARNING: {path} not found, skipping fig2")
        return

    df = pd.read_csv(path).head(15)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x="importance", y="feature", palette="viridis")
    plt.title("Top-15 Feature Importance (LightGBM)")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(FIGURES / "fig2_importance.png", dpi=150)
    plt.close()
    print("  INFO: fig2_importance.png saved")

def plot_final_metrics():
    """Plot: Comparison of Manual vs Manual+LLM on test."""
    path = EXPERIMENTS / "002_llm_rl" / "test_metrics_final.csv"
    if not path.exists():
        print(f"WARNING: {path} not found, skipping fig3")
        return

    df = pd.read_csv(path, index_col=0)
    metrics = ["NDCG@10", "Recall@10", "ROC-AUC"]
    
    # Keep only required metrics and transpose for barplot
    df_plot = df[metrics].T
    df_plot = df_plot.reset_index().rename(columns={"index": "Metric"})
    
    # Melt for seaborn
    df_melt = df_plot.melt(id_vars="Metric", var_name="Model", value_name="Score")

    plt.figure(figsize=(10, 5))
    sns.barplot(data=df_melt, x="Metric", y="Score", hue="Model", palette="muted")
    plt.title("Final Test Metrics: Manual vs Manual + Selected LLM")
    plt.ylim(0, 1.0)
    plt.legend(title="Model")
    plt.tight_layout()
    plt.savefig(FIGURES / "fig3_final_metrics.png", dpi=150)
    plt.close()
    print("  INFO: fig3_final_metrics.png saved")

if __name__ == "__main__":
    print("Generating plots...")
    plot_exhaustive()
    plot_feature_importance()
    plot_final_metrics()
    print(f"INFO: Plots saved to {FIGURES}/")