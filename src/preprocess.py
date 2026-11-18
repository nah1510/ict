"""Data preprocessing and sample session generator for Session-based Recommendation."""

import argparse
import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

CATEGORIES = [
    ("Điện thoại & Phụ kiện", ["iPhone 16", "Samsung S25", "Ốp lưng MagSafe", "Cáp sạc Type-C", "Sạc dự phòng 20000mAh", "Kính cường lực"]),
    ("Laptop & Văn phòng", ["MacBook Air M3", "ThinkPad X1", "Chuột Logitech MX", "Bàn phím cơ", "Balo chống sốc", "Đế tản nhiệt"]),
    ("Âm thanh", ["AirPods Pro", "Sony WH-1000XM5", "Loa Bluetooth JBL", "Micro thu âm", "Tai nghe gaming"]),
    ("Thời trang", ["Áo thun basic", "Quần jeans slim", "Giày sneaker trắng", "Áo khoác gió", "Mũ lưỡi trai"]),
    ("Sách & Học tập", ["Sách Deep Learning", "Sách Recommender Systems", "Sổ tay A5", "Bút ký cao cấp", "Đèn bàn LED"]),
]


def generate_sample_events(
    n_sessions: int = 4000,
    min_len: int = 2,
    max_len: int = 10,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Generate realistic synthetic browsing sessions with category-coherent transitions."""
    rng = np.random.default_rng(seed)
    items_meta = []
    cat_to_items: Dict[int, List[int]] = {}
    item_id = 1001
    for cat_idx, (cat_name, base_names) in enumerate(CATEGORIES):
        cat_to_items[cat_idx] = []
        for variant in range(1, 5):
            for name in base_names:
                items_meta.append(
                    {
                        "raw_item_id": item_id,
                        "item_name": f"{name} (Mẫu {variant})" if variant > 1 else name,
                        "category": cat_name,
                    }
                )
                cat_to_items[cat_idx].append(item_id)
                item_id += 1

    meta_df = pd.DataFrame(items_meta)
    records = []
    base_ts = 1_700_000_000

    for s_idx in range(n_sessions):
        sess_id = f"s_{s_idx:06d}"
        sess_len = int(rng.integers(min_len, max_len + 1))
        primary_cat = int(rng.integers(0, len(CATEGORIES)))
        cat_items = cat_to_items[primary_cat]

        # Start with an entry item in primary category, then follow sequential / co-occurrence pattern
        cur_pos = int(rng.integers(0, len(cat_items)))
        sess_ts = base_ts + s_idx * 120

        for step in range(sess_len):
            if rng.random() < 0.85:
                # Sequential step within the same category (forward or nearby item)
                cur_pos = (cur_pos + int(rng.choice([1, 1, 2, 0]))) % len(cat_items)
                chosen_item = cat_items[cur_pos]
            else:
                # Occasional cross-category browse
                other_cat = int(rng.integers(0, len(CATEGORIES)))
                chosen_item = int(rng.choice(cat_to_items[other_cat]))
            records.append(
                {
                    "session_id": sess_id,
                    "item_id": chosen_item,
                    "timestamp": sess_ts + step * int(rng.integers(5, 45)),
                }
            )

    events_df = pd.DataFrame(records)
    return events_df, meta_df


def filter_interactions(
    df: pd.DataFrame,
    min_item_freq: int = 5,
    min_session_len: int = 2,
) -> pd.DataFrame:
    """Iteratively filter rare items (< min_item_freq) and short sessions (< min_session_len)."""
    cleaned = df.sort_values(["session_id", "timestamp"]).copy()
    while True:
        prev_len = len(cleaned)
        item_counts = cleaned["item_id"].value_counts()
        valid_items = item_counts[item_counts >= min_item_freq].index
        cleaned = cleaned[cleaned["item_id"].isin(valid_items)]

        sess_counts = cleaned["session_id"].value_counts()
        valid_sessions = sess_counts[sess_counts >= min_session_len].index
        cleaned = cleaned[cleaned["session_id"].isin(valid_sessions)]

        if len(cleaned) == prev_len:
            break
    return cleaned.reset_index(drop=True)


def build_sessions_and_split(
    df: pd.DataFrame,
    valid_ratio: float = 0.1,
    test_ratio: float = 0.1,
) -> Tuple[List[List[int]], List[List[int]], List[List[int]], Dict[str, int]]:
    """Map items to 1..N, group by session, and split temporally by session start time."""
    unique_items = sorted(df["item_id"].unique().tolist())
    item2idx = {str(raw): idx + 1 for idx, raw in enumerate(unique_items)}

    df = df.copy()
    df["item_idx"] = df["item_id"].astype(str).map(item2idx)

    sess_groups = (
        df.sort_values("timestamp")
        .groupby("session_id", sort=False)
        .agg(items=("item_idx", list), start_ts=("timestamp", "min"))
        .sort_values("start_ts")
    )

    all_sessions = sess_groups["items"].tolist()
    n_total = len(all_sessions)
    n_test = max(1, int(n_total * test_ratio))
    n_valid = max(1, int(n_total * valid_ratio))
    n_train = n_total - n_valid - n_test

    train_sessions = all_sessions[:n_train]
    valid_sessions = all_sessions[n_train : n_train + n_valid]
    test_sessions = all_sessions[n_train + n_valid :]
    return train_sessions, valid_sessions, test_sessions, item2idx


def generate_prefix_pairs(sessions: List[List[int]]) -> List[Tuple[List[int], int]]:
    """Data augmentation: split each session [i1, i2, ..., it] into prefix->next pairs."""
    pairs: List[Tuple[List[int], int]] = []
    for sess in sessions:
        for t in range(1, len(sess)):
            pairs.append((sess[:t], sess[t]))
    return pairs


def save_processed(
    output_dir: Path,
    train_sessions: List[List[int]],
    valid_sessions: List[List[int]],
    test_sessions: List[List[int]],
    item2idx: Dict[str, int],
    meta_df: pd.DataFrame | None = None,
) -> None:
    """Persist processed sessions, item mapping, and optional item metadata."""
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "train_sessions.pkl", "wb") as f:
        pickle.dump(train_sessions, f)
    with open(output_dir / "valid_sessions.pkl", "wb") as f:
        pickle.dump(valid_sessions, f)
    with open(output_dir / "test_sessions.pkl", "wb") as f:
        pickle.dump(test_sessions, f)
    with open(output_dir / "item_map.json", "w", encoding="utf-8") as f:
        json.dump(item2idx, f, ensure_ascii=False, indent=2)

    if meta_df is not None:
        meta_out = meta_df.copy()
        meta_out["item_idx"] = meta_out["raw_item_id"].astype(str).map(item2idx)
        meta_out = meta_out.dropna(subset=["item_idx"]).copy()
        meta_out["item_idx"] = meta_out["item_idx"].astype(int)
        meta_out.to_csv(output_dir / "item_meta.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess session data for Session-based RS.")
    parser.add_argument("--sample", action="store_true", help="Generate synthetic sample dataset.")
    parser.add_argument("--input_csv", type=str, default=None, help="Path to CSV with session_id,item_id,timestamp.")
    parser.add_argument("--output_dir", type=str, default="data/processed", help="Output directory.")
    parser.add_argument("--min_item_freq", type=int, default=5)
    parser.add_argument("--min_session_len", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    meta_df = None
    if args.sample:
        print("Generating synthetic e-commerce session dataset...")
        events_df, meta_df = generate_sample_events(seed=args.seed)
    elif args.input_csv:
        print(f"Loading events from {args.input_csv}...")
        events_df = pd.read_csv(args.input_csv)
    else:
        raise ValueError("Provide either --sample or --input_csv.")

    raw_sessions = events_df["session_id"].nunique()
    raw_items = events_df["item_id"].nunique()

    cleaned_df = filter_interactions(
        events_df,
        min_item_freq=args.min_item_freq,
        min_session_len=args.min_session_len,
    )
    train_s, valid_s, test_s, item2idx = build_sessions_and_split(cleaned_df)
    save_processed(Path(args.output_dir), train_s, valid_s, test_s, item2idx, meta_df)

    print(f"Raw: {len(events_df)} events, {raw_sessions} sessions, {raw_items} items")
    print(f"Filtered: {len(cleaned_df)} events, {len(train_s)+len(valid_s)+len(test_s)} sessions, {len(item2idx)} items")
    print(f"Split -> Train: {len(train_s)}, Valid: {len(valid_s)}, Test: {len(test_s)}")
    print(f"Saved processed data to {args.output_dir}/")


if __name__ == "__main__":
    main()
