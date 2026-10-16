#!/usr/bin/env python3
"""Графики и числовая сводка по результатам bench (CSV из make bench-run).

Использование: plot.py <results.csv> [output_dir]

ИСПРАВЛЕНО (задача 2, "Графики"): скрипта построения графиков в репозитории
не было. Этот скрипт также не зависит от seaborn (в старой версии он
импортировался, но не был указан в requirements.txt, и скрипт падал на
чистой установке).

Ускорение считается относительно последовательного прогона ИЗ ТОГО ЖЕ CSV и
для той же серии (эксперимент, размер, фильтр, длина цепочки): одна и та же
базовая линия для всех точек серии, а не новая для каждой точки.

Кроме PNG пишется summary.md с таблицами чисел - выводы в README нужно
подтверждать этими числами, а не словами "похоже".
"""
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

SERIES = ["experiment", "size", "filter"]


def load(path):
    df = pd.read_csv(path)
    base = (
        df[df["mode"] == "seq"][SERIES + ["median_sec"]]
        .rename(columns={"median_sec": "seq_sec"})
    )
    df = df.merge(base, on=SERIES, how="left")
    df["speedup"] = df["seq_sec"] / df["median_sec"]
    df["efficiency"] = df["speedup"] / df["threads"]
    return df


def savefig(fig, out_dir, name):
    path = os.path.join(out_dir, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", path)


def line_panel(ax, sub, group, y, ylabel, ideal=False):
    for key, g in sub.groupby(group, sort=False):
        g = g.sort_values("threads")
        ax.plot(g["threads"], g[y], "o-", label=str(key))
    if ideal:
        ths = sorted(sub["threads"].unique())
        ax.plot(ths, ths, "k--", alpha=0.4, label="идеал")
    ax.set_xlabel("потоки")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)


def par(df, exp):
    return df[(df["experiment"] == exp) & (df["mode"] != "seq")]


def plot_strategies(df, out_dir):
    d = par(df, "strategies")
    if d.empty:
        return
    filters = list(dict.fromkeys(d["filter"]))
    sizes = sorted(d["size"].unique())
    fig, axes = plt.subplots(
        len(filters), len(sizes), figsize=(4 * len(sizes), 3.2 * len(filters)),
        squeeze=False)
    for i, f in enumerate(filters):
        for j, s in enumerate(sizes):
            ax = axes[i][j]
            line_panel(ax, d[(d["filter"] == f) & (d["size"] == s)], "mode",
                       "speedup", "ускорение", ideal=True)
            ax.set_title(f"{f}, {s}x{s}", fontsize=10)
    axes[0][0].legend(fontsize=8)
    fig.suptitle("Ускорение по стратегиям (строки - фильтры, столбцы - размеры)")
    fig.tight_layout()
    savefig(fig, out_dir, "01_strategies_speedup.png")

    big = d[d["size"] == sizes[-1]]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, y, label in ((axes[0], "efficiency", "эффективность (ускорение / потоки)"),
                         (axes[1], "mpix_per_sec", "МПикс/с")):
        sub = big[big["filter"] == filters[0]]
        line_panel(ax, sub, "mode", y, label)
        ax.set_title(f"{filters[0]}, {sizes[-1]}x{sizes[-1]}")
    axes[0].legend()
    savefig(fig, out_dir, "02_efficiency_throughput.png")

    # Влияние размера фильтра (3x3 / 5x5 / 9x9) на масштабирование.
    row = big[big["mode"] == "row"]
    if row["filter"].nunique() > 1:
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        line_panel(ax, row, "filter", "speedup", "ускорение (row)", ideal=True)
        ax.set_title(f"Влияние размера фильтра, {sizes[-1]}x{sizes[-1]}")
        ax.legend()
        savefig(fig, out_dir, "03_filter_size.png")


def plot_generic(df, out_dir, exp, group_col, name, title, per_mode=True):
    d = par(df, exp)
    if d.empty:
        return
    modes = list(dict.fromkeys(d["mode"])) if per_mode else [None]
    fig, axes = plt.subplots(1, len(modes), figsize=(6 * len(modes), 4.5),
                             squeeze=False)
    for ax, m in zip(axes[0], modes):
        sub = d if m is None else d[d["mode"] == m]
        line_panel(ax, sub, group_col, "speedup", "ускорение", ideal=True)
        ax.set_title(f"{title}" + (f" (режим {m})" if m else ""))
        ax.legend(fontsize=8)
    savefig(fig, out_dir, name)


def plot_chain(df, out_dir):
    d = df[df["experiment"] == "chain"]
    if d.empty:
        return
    max_t = int(d["threads"].max())
    fig, ax = plt.subplots(figsize=(7, 4.5))
    seq = d[d["mode"] == "seq"].sort_values("chain")
    ax.plot(seq["chain"], seq["median_sec"], "o-", label="seq")
    for m in ("row", "tile"):
        g = d[(d["mode"] == m) & (d["threads"] == max_t)].sort_values("chain")
        ax.plot(g["chain"], g["median_sec"], "s-", label=f"{m}, {max_t} потоков")
    ax.set_xlabel("число последовательно применённых фильтров")
    ax.set_ylabel("время, с")
    ax.set_title("Композиция фильтров")
    ax.grid(alpha=0.3)
    ax.legend()
    savefig(fig, out_dir, "06_chain.png")


def md_table(pivot):
    cols = list(pivot.columns)
    lines = ["| " + " | ".join([str(pivot.index.name)] + [str(c) for c in cols]) + " |",
             "|" + "---|" * (len(cols) + 1)]
    for idx, row in pivot.iterrows():
        cells = ["" if pd.isna(v) else f"{v:.2f}" for v in row]
        lines.append("| " + " | ".join([str(idx)] + cells) + " |")
    return "\n".join(lines)


def summary(df, out_dir):
    out = ["# Сводка бенчмарка (числа, из которых строятся выводы)\n"]
    seq = df[df["mode"] == "seq"]
    out.append("## Время последовательной версии (медиана, мс)\n")
    seq = seq.assign(median_ms=seq["median_sec"] * 1000.0)
    t = seq[seq["experiment"] == "strategies"].pivot_table(
        index="filter", columns="size", values="median_ms", sort=False)
    out.append(md_table(t))

    for exp, key, title in (
            ("strategies", "mode", "Ускорение по стратегиям"),
            ("tiles", "tile", "Ускорение по формам плиток"),
            ("schedules", "schedule", "Ускорение по политикам schedule")):
        d = par(df, exp)
        if d.empty:
            continue
        size = d["size"].max()
        for f in list(dict.fromkeys(d["filter"])):
            sub = d[(d["size"] == size) & (d["filter"] == f)].copy()
            if exp == "schedules":
                sub["key"] = sub["mode"] + "/" + sub["schedule"]
            else:
                sub["key"] = sub[key]
            piv = sub.pivot_table(index="key", columns="threads",
                                  values="speedup", sort=False)
            piv.index.name = key
            out.append(f"\n## {title}: {f}, {size}x{size}\n")
            out.append(md_table(piv))
            mt = sub["threads"].max()
            best = sub[sub["threads"] == mt].sort_values("speedup").iloc[-1]
            worst = sub[sub["threads"] == mt].sort_values("speedup").iloc[0]
            out.append(f"\nПри {mt} потоках лучший вариант: {best['key']} "
                       f"(x{best['speedup']:.2f}), худший: {worst['key']} "
                       f"(x{worst['speedup']:.2f}).")

    d = df[df["experiment"] == "chain"]
    if not d.empty:
        out.append("\n## Композиция фильтров: время, мс\n")
        mt = d["threads"].max()
        sel = d[(d["mode"] == "seq") | (d["threads"] == mt)].copy()
        sel["median_ms"] = sel["median_sec"] * 1000.0
        sel["key"] = sel["mode"].where(sel["mode"] == "seq",
                                       sel["mode"] + f"/{mt}t")
        piv = sel.pivot_table(index="key", columns="chain",
                              values="median_ms", sort=False)
        piv.index.name = "режим"
        out.append(md_table(piv))

    path = os.path.join(out_dir, "summary.md")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(out) + "\n")
    print("Saved:", path)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "benchmarks"
    os.makedirs(out_dir, exist_ok=True)
    df = load(sys.argv[1])
    plot_strategies(df, out_dir)
    plot_generic(df, out_dir, "tiles", "tile", "04_tiles.png",
                 "Форма плитки")
    plot_generic(df, out_dir, "schedules", "schedule", "05_schedules.png",
                 "schedule", per_mode=True)
    plot_chain(df, out_dir)
    summary(df, out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
