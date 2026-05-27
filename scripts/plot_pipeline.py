#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)

def load_csv(path):
    df = pd.read_csv(path)
    return df

def plot_speedup(df, out='plots/pipeline_speedup.png'):
    plt.figure(figsize=(10, 6))
    
    for mode in df['mode'].unique():
        sub = df[df['mode'] == mode]
        plt.plot(sub['workers'], sub['speedup'], 'o-', label=mode)
    
    plt.axhline(y=1, color='gray', linestyle='--', alpha=0.5, label='Baseline')
    plt.xlabel('Количество потоков конвейера')
    plt.ylabel('Ускорение (Speedup)')
    plt.title('Эффективность конвейерной обработки')
    plt.legend()
    plt.grid(alpha=0.3)
    
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"Saved: {out}")

def plot_time(df, out='plots/pipeline_time.png'):
    plt.figure(figsize=(10, 6))
    
    for mode in df['mode'].unique():
        sub = df[df['mode'] == mode]
        plt.plot(sub['workers'], sub['time_sec'], 's-', label=mode)
    
    plt.xlabel('Количество потоков конвейера')
    plt.ylabel('Время (сек)')
    plt.title('Время обработки набора изображений')
    plt.legend()
    plt.grid(alpha=0.3)
    
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"Saved: {out}")

def generate_table(df):
    print("\n### Сводная таблица результатов")
    print("| Workers | Mode | Time (s) | Speedup |")
    print("|---------|------|----------|---------|")
    for _, row in df.iterrows():
        print(f"| {row['workers']:7} | {row['mode']:4} | {row['time_sec']:8.3f} | {row['speedup']:7.2f}x |")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: plot_pipeline.py <pipeline.csv> [output_dir]")
        sys.exit(1)
    
    df = load_csv(sys.argv[1])
    out_dir = sys.argv[2] if len(sys.argv) > 2 else 'plots'
    
    os.makedirs(out_dir, exist_ok=True)
    plot_speedup(df, f'{out_dir}/pipeline_speedup.png')
    plot_time(df, f'{out_dir}/pipeline_time.png')
    generate_table(df)