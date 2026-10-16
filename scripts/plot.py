#!/usr/bin/env python3
"""Графики по результатам bench: plot.py <results.csv> [output_dir] [threads].

Графики (для каждого размера изображения 128/512/1024/2048):
  01_strategy_comparison.png - время разных стратегий при фиксированном
                               размере фильтра (3x3) и числе потоков threads;
  02_speedup_analysis.png    - ускорение относительно последовательной версии;
  03_efficiency.png          - эффективность (ускорение / потоки, %);
  04_filter_size_speedup.png - ускорение в зависимости от размера фильтра
                               (3x3, 5x5, 9x9) при threads потоках;
  05_tiles.png               - влияние формы и размера плитки (если есть).
Число потоков threads для графиков 01 и 04 по умолчанию - максимальное в CSV
(для 20 потоков: plot.py results.csv plots 20).

Что изменено по сравнению со старым скриптом:
  - читает новый формат CSV (размер - число, медиана вместо среднего);
  - ускорение считается от seq-строки ТОЙ ЖЕ серии (размер, фильтр) из того
    же CSV (раньше среднее по всем потокам смешивалось в одну точку);
  - ось потоков - реальные значения из CSV (раньше зашито [1, 2, 4, 8]);
  - не зависит от seaborn (его не было в requirements.txt);
  - пишет summary.md с числами: выводы должны опираться на них.
"""
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

try:
    plt.style.use("seaborn-v0_8-whitegrid")
except OSError:
    pass
plt.rcParams["font.size"] = 10

STRATEGIES = ["pixel", "row", "col", "tile"]
LABELS = {"pixel": "Pixel", "row": "Row", "col": "Col", "tile": "Tile"}
COLORS = {"pixel": "#FF6B6B", "row": "#4ECDC4", "col": "#45B7D1",
          "tile": "#FFA07A"}


def load_data(csv_file):
    df = pd.read_csv(csv_file)
    # Ускорение - от последовательной версии той же серии (размер, фильтр).
    key = ["experiment", "size", "filter"]
    seq = df[df["mode"] == "seq"][key + ["median_sec"]].rename(
        columns={"median_sec": "seq_sec"})
    df = df.merge(seq, on=key, how="left")
    df["speedup"] = df["seq_sec"] / df["median_sec"]
    df["efficiency"] = df["speedup"] / df["threads"] * 100.0
    return df


def size_grid(sizes):
    rows = (len(sizes) + 1) // 2
    fig, axes = plt.subplots(rows, 2, figsize=(14, 5 * rows), squeeze=False)
    axes = axes.flatten()
    for ax in axes[len(sizes):]:
        ax.axis("off")
    return fig, axes


def save(fig, out_dir, name, title):
    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    path = os.path.join(out_dir, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Создан график:", path)


def plot_strategy_comparison(df, base_fs, threads, out_dir):
    """01: время стратегий для каждого размера (фильтр base_fs x base_fs)."""
    sizes = sorted(df["size"].unique())
    fig, axes = size_grid(sizes)
    for ax, size in zip(axes, sizes):
        sub = df[(df["size"] == size) & (df["filter_size"] == base_fs)]
        times = [sub[(sub["mode"] == s) & (sub["threads"] == threads)]
                 ["median_sec"].mean() for s in STRATEGIES]
        times = [0 if pd.isna(t) else t for t in times]
        bars = ax.bar([LABELS[s] for s in STRATEGIES], times,
                      color=[COLORS[s] for s in STRATEGIES], alpha=0.7,
                      edgecolor="black")
        for bar, t in zip(bars, times):
            ax.text(bar.get_x() + bar.get_width() / 2, t, f"{t * 1000:.2f}ms",
                    ha="center", va="bottom", fontsize=9)
        seq = sub[sub["mode"] == "seq"]["median_sec"].mean()
        if not pd.isna(seq):
            ax.axhline(seq, color="gray", ls="--", alpha=0.7,
                       label=f"seq: {seq * 1000:.2f}ms")
            ax.legend(fontsize=8)
        ax.set_title(f"Размер: {size}x{size}", fontsize=12, fontweight="bold")
        ax.set_ylabel("Время (сек)")
        ax.grid(axis="y", alpha=0.3)
    save(fig, out_dir, "01_strategy_comparison.png",
         f"Время стратегий параллелизации (фильтр {base_fs}x{base_fs}, "
         f"{threads} потоков)")


def lines_by_size(df, base_fs, out_dir, name, title, y, ylabel, min_threads=1,
                  ref_lines=()):
    sizes = sorted(df["size"].unique())
    fig, axes = size_grid(sizes)
    for ax, size in zip(axes, sizes):
        sub = df[(df["size"] == size) & (df["filter_size"] == base_fs) &
                 (df["mode"] != "seq") & (df["threads"] >= min_threads)]
        for s in STRATEGIES:
            d = sub[sub["mode"] == s].sort_values("threads")
            ax.plot(d["threads"], d[y], marker="o", linewidth=2,
                    color=COLORS[s], label=LABELS[s])
        for value, color, style, label in ref_lines:
            ax.axhline(value, color=color, ls=style, alpha=0.5, label=label)
        if y == "speedup":
            ths = sorted(sub["threads"].unique())
            ax.plot(ths, ths, "k:", alpha=0.4, label="Идеал")
        ax.set_xticks(sorted(sub["threads"].unique()))
        ax.set_title(f"{title} для {size}x{size}", fontsize=12,
                     fontweight="bold")
        ax.set_xlabel("Количество потоков")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    return fig


def plot_speedup(df, base_fs, out_dir):
    """02: ускорение относительно последовательной версии."""
    fig = lines_by_size(
        df, base_fs, out_dir, "02", "Ускорение", "speedup", "Ускорение (раз)",
        ref_lines=[(1, "gray", "--", "Без ускорения")])
    save(fig, out_dir, "02_speedup_analysis.png",
         f"Ускорение относительно последовательной версии "
         f"(фильтр {base_fs}x{base_fs})")


def plot_efficiency(df, base_fs, out_dir):
    """03: эффективность = ускорение / потоки * 100%."""
    fig = lines_by_size(
        df, base_fs, out_dir, "03", "Эффективность", "efficiency",
        "Эффективность (%)", min_threads=2,
        ref_lines=[(100, "green", "--", "Идеальная (100%)"),
                   (50, "orange", ":", "50% эффективность")])
    save(fig, out_dir, "03_efficiency.png",
         "Эффективность параллелизации (Speedup / Threads × 100%)")


def plot_filter_size(df, threads, out_dir):
    """04: ускорение в зависимости от размера фильтра при threads потоках."""
    sizes = sorted(df["size"].unique())
    fig, axes = size_grid(sizes)
    for ax, size in zip(axes, sizes):
        sub = df[(df["size"] == size) & (df["threads"] == threads) &
                 (df["mode"] != "seq")]
        for s in STRATEGIES:
            d = sub[sub["mode"] == s].sort_values("filter_size")
            ax.plot(d["filter_size"], d["speedup"], marker="o", linewidth=2,
                    color=COLORS[s], label=LABELS[s])
        ax.axhline(1, color="gray", ls="--", alpha=0.5, label="Без ускорения")
        ax.set_xticks(sorted(sub["filter_size"].unique()))
        ax.set_xticklabels([f"{k}x{k}" for k in
                            sorted(sub["filter_size"].unique())])
        ax.set_title(f"Размер фильтра, {size}x{size}", fontsize=12,
                     fontweight="bold")
        ax.set_xlabel("Размер фильтра")
        ax.set_ylabel("Ускорение (раз)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    save(fig, out_dir, "04_filter_size_speedup.png",
         f"Ускорение в зависимости от размера фильтра ({threads} потоков)")


def plot_tiles(df, out_dir):
    """05: форма и размер плитки (эксперимент tiles)."""
    d = df[(df["experiment"] == "tiles") & (df["mode"] == "tile")]
    if d.empty:
        return
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for tile, g in d.groupby("tile", sort=False):
        g = g.sort_values("threads")
        ax.plot(g["threads"], g["speedup"], marker="o", label=tile)
    ths = sorted(d["threads"].unique())
    ax.plot(ths, ths, "k:", alpha=0.4, label="Идеал")
    ax.set_xticks(ths)
    ax.set_xlabel("Количество потоков")
    ax.set_ylabel("Ускорение (раз)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    size = int(d["size"].iloc[0])
    save(fig, out_dir, "05_tiles.png",
         f"Влияние формы и размера плитки ({size}x{size})")


def md(rows, header):
    lines = ["| " + " | ".join(header) + " |",
             "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(lines)


def summary(df, base_fs, threads, out_dir):
    out = ["# Сводка бенчмарка\n"]
    d = df[df["experiment"] == "strategies"]
    for size in sorted(d["size"].unique()):
        out.append(f"\n## {size}x{size}, фильтр {base_fs}x{base_fs}, "
                   f"{threads} потоков\n")
        sub = d[(d["size"] == size) & (d["filter_size"] == base_fs)]
        seq = sub[sub["mode"] == "seq"]["median_sec"].mean()
        rows = [("seq", f"{seq * 1000:.2f}", "1.00", "-")]
        for s in STRATEGIES:
            r = sub[(sub["mode"] == s) & (sub["threads"] == threads)]
            if r.empty:
                continue
            r = r.iloc[0]
            rows.append((s, f"{r.median_sec * 1000:.2f}",
                         f"{r.speedup:.2f}", f"{r.efficiency:.0f}%"))
        out.append(md(rows, ["стратегия", "время, мс", "ускорение",
                             "эффективность"]))

    out.append(f"\n## Ускорение от размера фильтра ({threads} потоков)\n")
    rows = []
    for size in sorted(d["size"].unique()):
        for s in STRATEGIES:
            cells = []
            for k in sorted(d["filter_size"].unique()):
                r = d[(d["size"] == size) & (d["mode"] == s) &
                      (d["threads"] == threads) & (d["filter_size"] == k)]
                cells.append(f"{r.iloc[0].speedup:.2f}" if len(r) else "-")
            rows.append((f"{size}x{size}", s, *cells))
    ks = [f"{k}x{k}" for k in sorted(d["filter_size"].unique())]
    out.append(md(rows, ["размер", "стратегия", *ks]))

    t = df[(df["experiment"] == "tiles") & (df["mode"] == "tile") &
           (df["threads"] == threads)]
    if not t.empty:
        out.append(f"\n## Плитки ({threads} потоков)\n")
        rows = [(r.tile, f"{r.median_sec * 1000:.2f}", f"{r.speedup:.2f}")
                for r in t.itertuples()]
        out.append(md(rows, ["плитка", "время, мс", "ускорение"]))

    path = os.path.join(out_dir, "summary.md")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(out) + "\n")
    print("Создана сводка:", path)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "benchmarks"
    os.makedirs(out_dir, exist_ok=True)
    df = load_data(sys.argv[1])
    strat = df[df["experiment"] == "strategies"]
    threads = int(sys.argv[3]) if len(sys.argv) > 3 else int(
        strat["threads"].max())
    base_fs = int(strat["filter_size"].min())

    plot_strategy_comparison(strat, base_fs, threads, out_dir)
    plot_speedup(strat, base_fs, out_dir)
    plot_efficiency(strat, base_fs, out_dir)
    plot_filter_size(strat, threads, out_dir)
    plot_tiles(df, out_dir)
    summary(df, base_fs, threads, out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())