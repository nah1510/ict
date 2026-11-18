"""End-to-end training, multi-seed evaluation, and ablation analysis for Session-based RS."""

import argparse
import json
import pickle
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from src.baselines import (
    BaseSessionRecommender,
    ItemKNNRecommender,
    PopularityRecommender,
    SessionKNNRecommender,
    SessionPopularityRecommender,
)
from src.gru4rec import GRU4RecRecommender
from src.metrics import evaluate_predictions
from src.preprocess import generate_prefix_pairs


def length_bucket(prefix_len: int) -> str:
    if prefix_len <= 2:
        return "Ngắn (1-2)"
    if prefix_len <= 5:
        return "Trung bình (3-5)"
    return "Dài (6+)"


def evaluate_model(
    model: BaseSessionRecommender,
    test_pairs: Sequence[Tuple[List[int], int]],
    k: int = 20,
) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]]]:
    """Evaluate a model overall and grouped by session prefix length."""
    overall_preds = []
    bucket_preds: Dict[str, List[Tuple[List[int], int]]] = {
        "Ngắn (1-2)": [],
        "Trung bình (3-5)": [],
        "Dài (6+)": [],
    }
    for prefix, target in test_pairs:
        recs = model.predict_next(prefix, top_k=k)
        overall_preds.append((recs, target))
        bucket_preds[length_bucket(len(prefix))].append((recs, target))

    overall_metrics = evaluate_predictions(overall_preds, k=k)
    bucket_metrics = {
        b: evaluate_predictions(preds, k=k) for b, preds in bucket_preds.items()
    }
    return overall_metrics, bucket_metrics


def plot_charts(
    bench_df: pd.DataFrame,
    len_df: pd.DataFrame,
    results_dir: Path,
    k: int,
) -> None:
    """Save comparison bar charts for slides and reports."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping chart generation.")
        return

    # 1. Benchmark chart
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    models = bench_df["Model"].tolist()
    hr_means = bench_df[f"HR@{k}_mean"].tolist()
    mrr_means = bench_df[f"MRR@{k}_mean"].tolist()

    axes[0].barh(models, hr_means, color="#2b6cb0")
    axes[0].set_title(f"So sánh HR@{k}")
    axes[0].set_xlabel(f"HR@{k}")
    axes[0].invert_yaxis()

    axes[1].barh(models, mrr_means, color="#2f855a")
    axes[1].set_title(f"So sánh MRR@{k}")
    axes[1].set_xlabel(f"MRR@{k}")
    axes[1].invert_yaxis()

    plt.tight_layout()
    plt.savefig(results_dir / "benchmark_chart.png", dpi=160)
    plt.close(fig)

    # 2. Session length chart
    if not len_df.empty:
        pivot = len_df.pivot(index="Bucket", columns="Model", values=f"HR@{k}")
        order = ["Ngắn (1-2)", "Trung bình (3-5)", "Dài (6+)"]
        pivot = pivot.reindex([o for o in order if o in pivot.index])
        ax = pivot.plot(kind="bar", figsize=(8, 4.5), rot=0)
        ax.set_title(f"HR@{k} theo độ dài tiền tố phiên")
        ax.set_ylabel(f"HR@{k}")
        ax.set_xlabel("Độ dài tiền tố phiên")
        plt.tight_layout()
        plt.savefig(results_dir / "session_length_chart.png", dpi=160)
        plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate Session-based RS models.")
    parser.add_argument("--data_dir", type=str, default="data/processed")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--eval_only", action="store_true", help="Print saved benchmark results.")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    if args.eval_only:
        bench_path = results_dir / "benchmark.csv"
        if not bench_path.exists():
            raise FileNotFoundError(f"No saved results found at {bench_path}")
        print(pd.read_csv(bench_path).to_string(index=False))
        return

    with open(data_dir / "train_sessions.pkl", "rb") as f:
        train_sessions = pickle.load(f)
    with open(data_dir / "test_sessions.pkl", "rb") as f:
        test_sessions = pickle.load(f)
    with open(data_dir / "item_map.json", "r", encoding="utf-8") as f:
        item_map = json.load(f)

    num_items = len(item_map)
    test_pairs = generate_prefix_pairs(test_sessions)
    print(f"Loaded {len(train_sessions)} train sessions, {len(test_pairs)} test prefix-target pairs, {num_items} items.")

    # Deterministic baselines (fit once)
    deterministic_models: List[BaseSessionRecommender] = [
        PopularityRecommender(),
        SessionPopularityRecommender(),
        ItemKNNRecommender(k_neighbors=100),
        SessionKNNRecommender(k_neighbors=100, recency_weighting=False),
        SessionKNNRecommender(k_neighbors=100, recency_weighting=True, decay=0.6),
    ]

    records = []
    len_records = []

    for model in deterministic_models:
        print(f"Training & evaluating {model.name}...")
        model.fit(train_sessions)
        overall, buckets = evaluate_model(model, test_pairs, k=args.k)
        records.append(
            {
                "Model": model.name,
                f"HR@{args.k}_mean": overall[f"HR@{args.k}"],
                f"HR@{args.k}_std": 0.0,
                f"MRR@{args.k}_mean": overall[f"MRR@{args.k}"],
                f"MRR@{args.k}_std": 0.0,
            }
        )
        if model.name in ("POP", "V-SKNN"):
            with open(results_dir / f"{model.name.lower().replace('-', '')}.pkl", "wb") as f:
                pickle.dump(model, f)
        if model.name == "V-SKNN":
            for b, m in buckets.items():
                len_records.append({"Model": model.name, "Bucket": b, f"HR@{args.k}": m[f"HR@{args.k}"], f"MRR@{args.k}": m[f"MRR@{args.k}"]})

    # Stochastic neural models (run across multiple seeds)
    for loss_type in ["ce", "bpr"]:
        hrs, mrrs = [], []
        last_model = None
        last_buckets = None
        for seed in args.seeds:
            print(f"Training GRU4Rec ({loss_type.upper()}) with seed={seed}...")
            model = GRU4RecRecommender(
                num_items=num_items,
                embed_dim=64,
                hidden_dim=100,
                epochs=args.epochs,
                loss_type=loss_type,
                seed=seed,
            )
            model.fit(train_sessions)
            overall, buckets = evaluate_model(model, test_pairs, k=args.k)
            hrs.append(overall[f"HR@{args.k}"])
            mrrs.append(overall[f"MRR@{args.k}"])
            last_model = model
            last_buckets = buckets

        records.append(
            {
                "Model": f"GRU4Rec ({loss_type.upper()})",
                f"HR@{args.k}_mean": float(np.mean(hrs)),
                f"HR@{args.k}_std": float(np.std(hrs)),
                f"MRR@{args.k}_mean": float(np.mean(mrrs)),
                f"MRR@{args.k}_std": float(np.std(mrrs)),
            }
        )
        if last_model is not None and loss_type == "ce":
            last_model.save(str(results_dir / "gru4rec_ce.pt"))
        if last_buckets is not None:
            for b, m in last_buckets.items():
                len_records.append({"Model": f"GRU4Rec ({loss_type.upper()})", "Bucket": b, f"HR@{args.k}": m[f"HR@{args.k}"], f"MRR@{args.k}": m[f"MRR@{args.k}"]})

    bench_df = pd.DataFrame(records)
    len_df = pd.DataFrame(len_records)
    bench_df.to_csv(results_dir / "benchmark.csv", index=False)
    len_df.to_csv(results_dir / "session_length_analysis.csv", index=False)

    plot_charts(bench_df, len_df, results_dir, k=args.k)
    print("\n=== BENCHMARK RESULTS ===")
    print(bench_df.to_string(index=False))


if __name__ == "__main__":
    main()
