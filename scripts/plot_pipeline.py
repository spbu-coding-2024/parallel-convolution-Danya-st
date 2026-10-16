#!/usr/bin/env python3
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

# Что изменено по сравнению со старым скриптом:
#  - убран "import seaborn": он не использовался и не был указан в
#    requirements.txt, поэтому скрипт падал на чистой установке;
#  - читает новый CSV bench_stream (одна baseline на весь файл, разбивка по
#    стадиям, простои на очередях);
#  - потолок ускорения и узкое место считаются из измерений, а не гадаются
#    ("ограничения в ядрах и накладные расходы" ничем не подтверждались);
#  - добавлены график стоимости стадий и сводка pipeline_summary.md.
try:
    plt.style.use("seaborn-v0_8-whitegrid")
except OSError:
    pass
plt.rcParams["figure.figsize"] = (12, 8)


def load_csv(path):
    df = pd.read_csv(path)
    return df


def plot_speedup(df, out="plots/pipeline_speedup.png"):
    plt.figure(figsize=(10, 6))

    for mode in df["mode"].unique():
        sub = df[df["mode"] == mode].sort_values("workers")
        plt.plot(sub["workers"], sub["speedup"], "o-", label=mode)

    plt.axhline(y=1, color="gray", linestyle="--", alpha=0.5, label="Baseline")
    plt.xlabel("Количество потоков конвейера")
    plt.ylabel("Ускорение (Speedup)")
    plt.title("Эффективность конвейерной обработки")
    plt.legend()
    plt.grid(alpha=0.3)

    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def plot_time(df, out="plots/pipeline_time.png"):
    plt.figure(figsize=(10, 6))

    for mode in df["mode"].unique():
        sub = df[df["mode"] == mode].sort_values("workers")
        plt.plot(sub["workers"], sub["time_sec"], "s-", label=mode)

    plt.axhline(y=df["baseline_sec"].iloc[0], color="gray", linestyle="--",
                alpha=0.5, label="Baseline (1 воркер, seq)")
    plt.xlabel("Количество потоков конвейера")
    plt.ylabel("Время (сек)")
    plt.title("Время обработки набора изображений")
    plt.legend()
    plt.grid(alpha=0.3)

    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def reference_stages(df):
    """Стоимость стадий при 1 воркере и последовательной свёртке."""
    seq = df[(df["mode"] == "seq")].sort_values("workers")
    return seq.iloc[0] if not seq.empty else df.sort_values("workers").iloc[0]


def plot_stages(df, out="plots/pipeline_stages.png"):
    """Стоимость стадий и потолок ускорения (закон Амдала для конвейера)."""
    ref = reference_stages(df)
    base = df["baseline_sec"].iloc[0]
    stages = [("чтение (декодирование)", ref["read_busy"]),
              ("свёртка", ref["conv_busy"]),
              ("запись (кодирование PNG)", ref["write_busy"])]

    fig, ax = plt.subplots(figsize=(7, 5.5))
    bottom = 0
    for label, value in stages:
        ax.bar(["стадии"], [value], bottom=bottom,
               label=f"{label}: {value:.3f} с")
        bottom += value
    # Читатель и писатель работают одновременно, поэтому нижняя граница
    # времени прогона - самая тяжёлая ОДНОПОТОЧНАЯ стадия.
    heavy = max(ref["read_busy"], ref["write_busy"])
    ax.axhline(heavy, color="red", ls="--",
               label=f"нижняя граница времени конвейера: {heavy:.3f} с "
                     f"(потолок ускорения ~x{base / heavy:.2f})")
    ax.set_ylabel("процессорное время, с (сумма по потокам)")
    ax.set_title("Стоимость стадий конвейера")
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.08))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def generate_table(df, out_md=None):
    lines = ["### Сводная таблица результатов",
             "| Workers | Mode | Time (s) | Speedup | Чтение | Свёртка | Запись |"
             " Читатель ждёт | Воркеры ждут данные | Писатель ждёт |",
             "|---------|------|----------|---------|--------|---------|--------|"
             "---------------|---------------------|---------------|"]
    for _, r in df.iterrows():
        lines.append(
            f"| {int(r['workers']):7} | {r['mode']:4} | {r['time_sec']:8.3f} | "
            f"{r['speedup']:7.2f}x | {r['read_busy']:6.3f} | "
            f"{r['conv_busy']:7.3f} | {r['write_busy']:6.3f} | "
            f"{r['reader_blocked']:13.3f} | {r['workers_starved']:19.3f} | "
            f"{r['writer_starved']:13.3f} |")

    ref = reference_stages(df)
    base = df["baseline_sec"].iloc[0]
    heavy = max(ref["read_busy"], ref["write_busy"])
    which = ("запись (кодирование PNG)" if ref["write_busy"] >= ref["read_busy"]
             else "чтение (декодирование)")
    lines.append(
        f"\nBaseline (1 воркер, seq): {base:.3f} с. Стоимость стадий при 1 "
        f"воркере: чтение {ref['read_busy']:.3f} с, свёртка "
        f"{ref['conv_busy']:.3f} с, запись {ref['write_busy']:.3f} с. "
        f"Самая тяжёлая однопоточная стадия: {which}, {heavy:.3f} с. Даже при "
        f"бесконечно быстрой свёртке и любом числе воркеров время прогона не "
        f"станет меньше ~{heavy:.3f} с, т.е. потолок ускорения ~x"
        f"{base / heavy:.2f} (закон Амдала для конвейера). Лучший результат "
        f"в этом прогоне: x{df['speedup'].max():.2f}.")
    text = "\n".join(lines)
    print("\n" + text)
    if out_md:
        with open(out_md, "w", encoding="utf-8") as fh:
            fh.write(text + "\n")
        print(f"Saved: {out_md}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: plot_pipeline.py <pipeline.csv> [output_dir]")
        sys.exit(1)

    df = load_csv(sys.argv[1])
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "plots"

    os.makedirs(out_dir, exist_ok=True)
    plot_speedup(df, f"{out_dir}/pipeline_speedup.png")
    plot_time(df, f"{out_dir}/pipeline_time.png")
    plot_stages(df, f"{out_dir}/pipeline_stages.png")
    generate_table(df, f"{out_dir}/pipeline_summary.md")