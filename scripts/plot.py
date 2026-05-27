import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sys
import os
from matplotlib.gridspec import GridSpec

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

def load_data(csv_file):
    """Загрузка данных из CSV"""
    df = pd.read_csv(csv_file)
    df['width'] = df['size'].apply(lambda x: int(x.split('x')[0]))
    df['height'] = df['size'].apply(lambda x: int(x.split('x')[1]))
    df['total_pixels'] = df['width'] * df['height']
    return df

def plot_strategy_comparison(df, output_dir='plots'):
    """График 1: Сравнение стратегий для каждого размера"""
    os.makedirs(output_dir, exist_ok=True)
    
    sizes = sorted(df['size'].unique())
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    strategies = ['pixel', 'row', 'col', 'tile']
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
    
    for idx, size in enumerate(sizes):
        if idx >= 4:
            break
        ax = axes[idx]
        subset = df[df['size'] == size]
        
        avg_times = subset.groupby('mode')['time_sec'].mean()
        
        bars = ax.bar(strategies, [avg_times.get(s, 0) for s in strategies], 
                     color=colors, alpha=0.7, edgecolor='black')
        
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height*1000:.1f}ms', ha='center', va='bottom', fontsize=9)
        
        ax.set_title(f'Размер: {size}', fontsize=12, fontweight='bold')
        ax.set_ylabel('Время (сек)')
        ax.set_xticklabels(['Pixel', 'Row', 'Col', 'Tile'], rotation=0)
        ax.grid(axis='y', alpha=0.3)
    
    plt.suptitle('Сравнение времени выполнения для разных стратегий параллелизации', 
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/01_strategy_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f" Создан график: {output_dir}/01_strategy_comparison.png")

def plot_speedup_vs_sequential(df, output_dir='plots'):
    """График 2: Ускорение относительно последовательной версии"""
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    sizes = sorted(df['size'].unique())
    strategies = ['pixel', 'row', 'col', 'tile']
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
    
    for idx, size in enumerate(sizes):
        if idx >= 4:
            break
        ax = axes[idx]
        subset = df[df['size'] == size]
        
        seq_time = subset[(subset['mode'] == 'seq') & (subset['threads'] == 1)]['time_sec'].mean()
        
        if pd.isna(seq_time) or seq_time == 0:
            continue
            
        speedup_data = []
        for strategy in strategies:
            strat_subset = subset[subset['mode'] == strategy]
            for threads in sorted(strat_subset['threads'].unique()):
                time_val = strat_subset[strat_subset['threads'] == threads]['time_sec'].mean()
                if not pd.isna(time_val) and time_val > 0:
                    speedup = seq_time / time_val
                    speedup_data.append({
                        'strategy': strategy,
                        'threads': threads,
                        'speedup': speedup
                    })
        
        if speedup_data:
            speedup_df = pd.DataFrame(speedup_data)
            
            for strategy in strategies:
                strat_data = speedup_df[speedup_df['strategy'] == strategy]
                color = colors[strategies.index(strategy)]
                ax.plot(strat_data['threads'], strat_data['speedup'], 
                       marker='o', label=strategy.capitalize(), linewidth=2, color=color)
        
        ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5, label='Без ускорения')
        ax.set_title(f'Ускорение для {size}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Количество потоков')
        ax.set_ylabel('Ускорение (раз)')
        ax.set_xticks([1, 2, 4, 8])
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Ускорение относительно последовательной версии', 
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/02_speedup_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f" Создан график: {output_dir}/02_speedup_analysis.png")

def plot_throughput(df, output_dir='plots'):
    """График 3: Пропускная способность (MPix/s)"""
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    sizes = sorted(df['size'].unique())
    strategies = ['pixel', 'row', 'col', 'tile']
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
    
    for idx, size in enumerate(sizes):
        if idx >= 4:
            break
        ax = axes[idx]
        subset = df[df['size'] == size]
        
        throughput_data = []
        for strategy in strategies:
            strat_subset = subset[subset['mode'] == strategy]
            for threads in sorted(strat_subset['threads'].unique()):
                time_val = strat_subset[strat_subset['threads'] == threads]['time_sec'].mean()
                if not pd.isna(time_val) and time_val > 0:
                    pixels = subset['total_pixels'].iloc[0]
                    throughput = (pixels / time_val) / 1e6  # MPix/s
                    throughput_data.append({
                        'strategy': strategy,
                        'threads': threads,
                        'throughput': throughput
                    })
        
        if throughput_data:
            throughput_df = pd.DataFrame(throughput_data)
            
            for strategy in strategies:
                strat_data = throughput_df[throughput_df['strategy'] == strategy]
                color = colors[strategies.index(strategy)]
                ax.plot(strat_data['threads'], strat_data['throughput'], 
                       marker='s', label=strategy.capitalize(), linewidth=2, color=color)
        
        ax.set_title(f'Пропускная способность для {size}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Количество потоков')
        ax.set_ylabel('Пропускная способность (MPix/s)')
        ax.set_xticks([1, 2, 4, 8])
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Пропускная способность обработки изображений', 
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/03_throughput.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Создан график: {output_dir}/03_throughput.png")

def plot_average_times(df, output_dir='plots'):
    """График 4: Среднее время для каждой стратегии"""
    os.makedirs(output_dir, exist_ok=True)
    
    avg_df = df.groupby(['size', 'mode'])['time_sec'].mean().reset_index()
    
    plt.figure(figsize=(12, 6))
    
    x = np.arange(len(avg_df['size'].unique()))
    width = 0.2
    
    strategies = ['pixel', 'row', 'col', 'tile']
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
    
    for i, strategy in enumerate(strategies):
        strat_data = avg_df[avg_df['mode'] == strategy]['time_sec'].values * 1000  # в мс
        plt.bar(x + i*width, strat_data, width, label=strategy.capitalize(), 
               color=colors[i], alpha=0.8, edgecolor='black')
    
    plt.xlabel('Размер изображения', fontsize=12)
    plt.ylabel('Среднее время (мс)', fontsize=12)
    plt.title('Среднее время выполнения для разных стратегий', fontsize=14, fontweight='bold')
    plt.xticks(x + width*1.5, avg_df['size'].unique(), rotation=0)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/04_average_times.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Создан график: {output_dir}/04_average_times.png")

def plot_efficiency(df, output_dir='plots'):
    """График 5: Эффективность параллелизации"""
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    sizes = sorted(df['size'].unique())
    strategies = ['pixel', 'row', 'col', 'tile']
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
    
    for idx, size in enumerate(sizes):
        if idx >= 4:
            break
        ax = axes[idx]
        subset = df[df['size'] == size]
        
        best_1thread = subset[subset['threads'] == 1]['time_sec'].min()
        
        if pd.isna(best_1thread) or best_1thread == 0:
            continue
        
        efficiency_data = []
        for strategy in strategies:
            strat_subset = subset[subset['mode'] == strategy]
            for threads in [2, 4, 8]:
                time_val = strat_subset[strat_subset['threads'] == threads]['time_sec'].mean()
                if not pd.isna(time_val) and time_val > 0:
                    speedup = best_1thread / time_val
                    efficiency = (speedup / threads) * 100  # в процентах
                    efficiency_data.append({
                        'strategy': strategy,
                        'threads': threads,
                        'efficiency': efficiency
                    })
        
        if efficiency_data:
            eff_df = pd.DataFrame(efficiency_data)
            
            for strategy in strategies:
                strat_data = eff_df[eff_df['strategy'] == strategy]
                color = colors[strategies.index(strategy)]
                ax.plot(strat_data['threads'], strat_data['efficiency'], 
                       marker='o', label=strategy.capitalize(), linewidth=2, color=color)
        
        ax.axhline(y=100, color='green', linestyle='--', alpha=0.5, label='Идеальная (100%)')
        ax.axhline(y=50, color='orange', linestyle=':', alpha=0.5, label='50% эффективность')
        ax.set_title(f'Эффективность для {size}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Количество потоков')
        ax.set_ylabel('Эффективность (%)')
        ax.set_xticks([2, 4, 8])
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Эффективность параллелизации (Speedup / Threads × 100%)', 
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/05_efficiency.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Создан график: {output_dir}/05_efficiency.png")

def generate_summary_table(df):
    """Генерация сводной таблицы"""
    print("\n" + "="*80)
    print("СВОДНАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ")
    print("="*80)
    
    sizes = sorted(df['size'].unique())
    strategies = ['pixel', 'row', 'col', 'tile']
    
    for size in sizes:
        print(f"\nРазмер: {size}")
        print("-" * 80)
        print(f"{'Стратегия':<10} {'Среднее время (мс)':<20} {'Ускорение':<15} {'Эффективность':<15}")
        print("-" * 80)
        
        subset = df[df['size'] == size]
        seq_time = subset[(subset['mode'] == 'seq') & (subset['threads'] == 1)]['time_sec'].mean()
        
        for strategy in strategies:
            strat_subset = subset[subset['mode'] == strategy]
            best_time = strat_subset[strat_subset['threads'] == 8]['time_sec'].mean()
            
            if pd.isna(best_time) or best_time == 0 or pd.isna(seq_time) or seq_time == 0:
                continue
                
            speedup = seq_time / best_time
            efficiency = (speedup / 8) * 100
            
            print(f"{strategy:<10} {best_time*1000:<20.2f} {speedup:<15.2f} {efficiency:<15.1f}%")


def main():
    if len(sys.argv) < 2:
        print("Использование: python plot_advanced.py <results.csv> [output_dir]")
        sys.exit(1)
    
    csv_file = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else 'plots'
    
    print("Загрузка данных...")
    df = load_data(csv_file)
    
    print("Генерация графиков...")
    plot_strategy_comparison(df, output_dir)
    plot_speedup_vs_sequential(df, output_dir)
    plot_throughput(df, output_dir)
    plot_average_times(df, output_dir)
    plot_efficiency(df, output_dir)
    
    print("\nГенерация сводной таблицы и выводов...")
    generate_summary_table(df)
    
    print(f"\nВсе графики сохранены в папке: {output_dir}/")
    print("="*80)

if __name__ == '__main__':
    main()