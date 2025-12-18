"""
EpiGraphNet - Sonuç Görselleştirme
Kaydedilmiş JSON sonuçlarından grafikler oluşturur.

Kullanım:
    python visualize_results.py --results all_experiments_results.json
    python visualize_results.py --results all_experiments_results.json --output-dir my_figures
"""

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Matplotlib ayarları
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.dpi'] = 100

# Makale sonuçları (referans)
PAPER_RESULTS = {
    "binary": {
        "CNN-LSTM": {"accuracy": 99.30, "recall": 98.69, "precision": 98.70, "f1": 98.62},
        "1D-CNN-LSTM": {"accuracy": 99.04, "recall": 97.69, "precision": 99.07, "f1": 98.30},
        "EpiGraphNet_DE(a=50)": {"accuracy": 99.30, "recall": 98.71, "precision": 99.12, "f1": 98.89},
        "EpiGraphNet_DE(a=25)": {"accuracy": 99.56, "recall": 99.31, "precision": 99.45, "f1": 99.36},
        "EpiGraphNet_BE(a=50)": {"accuracy": 99.00, "recall": 97.68, "precision": 98.96, "f1": 98.22},
        "EpiGraphNet_BE(a=25)": {"accuracy": 99.12, "recall": 98.15, "precision": 99.15, "f1": 98.63},
    },
    "multiclass": {
        "CNN-LSTM": {"accuracy": 77.70, "recall": 77.49, "precision": 78.70, "f1": 76.97},
        "1D-CNN-LSTM": {"accuracy": 79.50, "recall": 79.55, "precision": 79.96, "f1": 78.91},
        "EpiGraphNet_DE(a=50)": {"accuracy": 80.60, "recall": 80.28, "precision": 81.26, "f1": 79.60},
        "EpiGraphNet_DE(a=25)": {"accuracy": 82.47, "recall": 81.96, "precision": 82.13, "f1": 81.33},
        "EpiGraphNet_BE(a=50)": {"accuracy": 81.26, "recall": 80.87, "precision": 81.30, "f1": 80.34},
        "EpiGraphNet_BE(a=25)": {"accuracy": 82.51, "recall": 81.98, "precision": 82.25, "f1": 82.14},
    }
}


def plot_accuracy_comparison(all_results: dict, paper_results: dict, output_dir: str):
    """Model doğruluklarını karşılaştıran bar chart."""
    os.makedirs(output_dir, exist_ok=True)
    
    for class_name in all_results["experiments"]:
        class_label = "İkili Sınıflandırma" if class_name == "binary" else "Çok Sınıflı Sınıflandırma"
        
        models = list(all_results["experiments"][class_name].keys())
        our_acc = [all_results["experiments"][class_name][m]["accuracy"] for m in models]
        paper_acc = [paper_results.get(class_name, {}).get(m, {}).get("accuracy", 0) for m in models]
        
        x = np.arange(len(models))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(14, 7))
        
        bars1 = ax.bar(x - width/2, paper_acc, width, label='Makale', color='#2196F3', alpha=0.8)
        bars2 = ax.bar(x + width/2, our_acc, width, label='Bizim', color='#4CAF50', alpha=0.8)
        
        ax.set_ylabel('Doğruluk (%)')
        ax.set_title(f'{class_label} - Model Karşılaştırması')
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.legend()
        ax.set_ylim([0, 105])
        
        # Değerleri bar üstüne yaz
        for bar in bars1:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)
        
        for bar in bars2:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        filename = f"{output_dir}/accuracy_comparison_{class_name}.png"
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Kaydedildi: {filename}")


def plot_metrics_heatmap(all_results: dict, output_dir: str):
    """Tüm metrikleri gösteren heatmap."""
    os.makedirs(output_dir, exist_ok=True)
    
    for class_name in all_results["experiments"]:
        class_label = "İkili Sınıflandırma" if class_name == "binary" else "Çok Sınıflı Sınıflandırma"
        
        models = list(all_results["experiments"][class_name].keys())
        metrics = ["accuracy", "recall", "precision", "f1"]
        metric_labels = ["Doğruluk", "Duyarlılık", "Kesinlik", "F1"]
        
        data = []
        for model in models:
            row = [all_results["experiments"][class_name][model].get(m, 0) for m in metrics]
            data.append(row)
        
        data = np.array(data)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        sns.heatmap(
            data, 
            annot=True, 
            fmt='.2f',
            cmap='RdYlGn',
            xticklabels=metric_labels,
            yticklabels=models,
            vmin=0,
            vmax=100,
            ax=ax,
            linewidths=0.5,
            cbar_kws={'label': 'Değer (%)'}
        )
        
        ax.set_title(f'{class_label} - Metrik Değerleri')
        plt.tight_layout()
        filename = f"{output_dir}/metrics_heatmap_{class_name}.png"
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Kaydedildi: {filename}")


def plot_model_radar(all_results: dict, output_dir: str):
    """Her model için radar/spider chart."""
    os.makedirs(output_dir, exist_ok=True)
    
    metrics = ["accuracy", "recall", "precision", "f1"]
    metric_labels = ["Doğruluk", "Duyarlılık", "Kesinlik", "F1"]
    
    for class_name in all_results["experiments"]:
        class_label = "İkili Sınıflandırma" if class_name == "binary" else "Çok Sınıflı Sınıflandırma"
        
        models = list(all_results["experiments"][class_name].keys())
        n_models = len(models)
        
        # Renk paleti
        colors = sns.color_palette("husl", n_models)
        
        # Radar chart için açılar
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # Kapatmak için
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
        
        for i, model in enumerate(models):
            values = [all_results["experiments"][class_name][model].get(m, 0) for m in metrics]
            values += values[:1]  # Kapatmak için
            
            ax.plot(angles, values, 'o-', linewidth=2, label=model, color=colors[i])
            ax.fill(angles, values, alpha=0.1, color=colors[i])
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metric_labels)
        ax.set_ylim(0, 100)
        ax.set_title(f'{class_label} - Model Performansları', y=1.08)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        
        plt.tight_layout()
        filename = f"{output_dir}/radar_chart_{class_name}.png"
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Kaydedildi: {filename}")


def plot_difference_chart(all_results: dict, paper_results: dict, output_dir: str):
    """Makale ile aramızdaki farkı gösteren chart."""
    os.makedirs(output_dir, exist_ok=True)
    
    for class_name in all_results["experiments"]:
        class_label = "İkili Sınıflandırma" if class_name == "binary" else "Çok Sınıflı Sınıflandırma"
        
        models = list(all_results["experiments"][class_name].keys())
        differences = []
        
        for model in models:
            our_acc = all_results["experiments"][class_name][model]["accuracy"]
            paper_acc = paper_results.get(class_name, {}).get(model, {}).get("accuracy", our_acc)
            differences.append(our_acc - paper_acc)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        colors = ['#4CAF50' if d >= 0 else '#F44336' for d in differences]
        bars = ax.barh(models, differences, color=colors, alpha=0.8)
        
        ax.axvline(x=0, color='black', linewidth=0.8)
        ax.set_xlabel('Fark (Bizim - Makale) %')
        ax.set_title(f'{class_label} - Makale ile Karşılaştırma')
        
        # Değerleri bar yanına yaz
        for bar, diff in zip(bars, differences):
            width = bar.get_width()
            ax.annotate(f'{diff:+.2f}%',
                        xy=(width, bar.get_y() + bar.get_height()/2),
                        xytext=(5 if width >= 0 else -5, 0),
                        textcoords="offset points",
                        ha='left' if width >= 0 else 'right',
                        va='center', fontsize=10)
        
        plt.tight_layout()
        filename = f"{output_dir}/difference_chart_{class_name}.png"
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Kaydedildi: {filename}")


def plot_binary_vs_multiclass(all_results: dict, output_dir: str):
    """Binary ve Multi-class sonuçlarını karşılaştıran grouped bar chart."""
    os.makedirs(output_dir, exist_ok=True)
    
    if "binary" not in all_results["experiments"] or "multiclass" not in all_results["experiments"]:
        print("  ⚠ Binary veya multiclass sonuçları eksik, bu grafik atlanıyor.")
        return
    
    models = list(all_results["experiments"]["binary"].keys())
    binary_acc = [all_results["experiments"]["binary"][m]["accuracy"] for m in models]
    multi_acc = [all_results["experiments"]["multiclass"][m]["accuracy"] for m in models]
    
    x = np.arange(len(models))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    bars1 = ax.bar(x - width/2, binary_acc, width, label='İkili Sınıflandırma', color='#3F51B5', alpha=0.8)
    bars2 = ax.bar(x + width/2, multi_acc, width, label='Çok Sınıflı', color='#FF9800', alpha=0.8)
    
    ax.set_ylabel('Doğruluk (%)')
    ax.set_title('İkili vs Çok Sınıflı Sınıflandırma Karşılaştırması')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim([0, 105])
    
    # Değerleri bar üstüne yaz
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)
    
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    filename = f"{output_dir}/binary_vs_multiclass.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Kaydedildi: {filename}")


def plot_std_errorbar(all_results: dict, output_dir: str):
    """Standart sapma ile error bar grafiği."""
    os.makedirs(output_dir, exist_ok=True)
    
    for class_name in all_results["experiments"]:
        class_label = "İkili Sınıflandırma" if class_name == "binary" else "Çok Sınıflı Sınıflandırma"
        
        models = list(all_results["experiments"][class_name].keys())
        accuracies = []
        stds = []
        
        for model in models:
            acc = all_results["experiments"][class_name][model]["accuracy"]
            std = all_results["experiments"][class_name][model].get("accuracy_std", 0)
            accuracies.append(acc)
            stds.append(std)
        
        fig, ax = plt.subplots(figsize=(12, 7))
        
        colors = sns.color_palette("husl", len(models))
        bars = ax.bar(models, accuracies, yerr=stds, capsize=5, color=colors, alpha=0.8, 
                      error_kw={'elinewidth': 2, 'capthick': 2})
        
        ax.set_ylabel('Doğruluk (%)')
        ax.set_xlabel('Model')
        ax.set_title(f'{class_label} - Doğruluk (±Std)')
        ax.set_ylim([0, 105])
        plt.xticks(rotation=45, ha='right')
        
        # Değerleri bar üstüne yaz
        for bar, acc, std in zip(bars, accuracies, stds):
            height = bar.get_height()
            ax.annotate(f'{acc:.1f}±{std:.1f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height + std + 1),
                        ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        filename = f"{output_dir}/accuracy_with_std_{class_name}.png"
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Kaydedildi: {filename}")


def create_summary_figure(all_results: dict, paper_results: dict, output_dir: str):
    """Tek bir özet figür oluşturur."""
    os.makedirs(output_dir, exist_ok=True)
    
    fig = plt.figure(figsize=(20, 12))
    
    # 2x2 grid
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.25)
    
    # 1. Binary doğruluk karşılaştırması
    if "binary" in all_results["experiments"]:
        ax1 = fig.add_subplot(gs[0, 0])
        models = list(all_results["experiments"]["binary"].keys())
        our_acc = [all_results["experiments"]["binary"][m]["accuracy"] for m in models]
        paper_acc = [paper_results.get("binary", {}).get(m, {}).get("accuracy", 0) for m in models]
        
        x = np.arange(len(models))
        width = 0.35
        ax1.bar(x - width/2, paper_acc, width, label='Makale', color='#2196F3', alpha=0.8)
        ax1.bar(x + width/2, our_acc, width, label='Bizim', color='#4CAF50', alpha=0.8)
        ax1.set_ylabel('Doğruluk (%)')
        ax1.set_title('İkili Sınıflandırma')
        ax1.set_xticks(x)
        ax1.set_xticklabels(models, rotation=45, ha='right', fontsize=8)
        ax1.legend()
        ax1.set_ylim([0, 105])
    
    # 2. Multiclass doğruluk karşılaştırması
    if "multiclass" in all_results["experiments"]:
        ax2 = fig.add_subplot(gs[0, 1])
        models = list(all_results["experiments"]["multiclass"].keys())
        our_acc = [all_results["experiments"]["multiclass"][m]["accuracy"] for m in models]
        paper_acc = [paper_results.get("multiclass", {}).get(m, {}).get("accuracy", 0) for m in models]
        
        x = np.arange(len(models))
        width = 0.35
        ax2.bar(x - width/2, paper_acc, width, label='Makale', color='#2196F3', alpha=0.8)
        ax2.bar(x + width/2, our_acc, width, label='Bizim', color='#4CAF50', alpha=0.8)
        ax2.set_ylabel('Doğruluk (%)')
        ax2.set_title('Çok Sınıflı Sınıflandırma')
        ax2.set_xticks(x)
        ax2.set_xticklabels(models, rotation=45, ha='right', fontsize=8)
        ax2.legend()
        ax2.set_ylim([0, 105])
    
    # 3. Fark grafiği (Binary)
    if "binary" in all_results["experiments"]:
        ax3 = fig.add_subplot(gs[1, 0])
        models = list(all_results["experiments"]["binary"].keys())
        differences = []
        for model in models:
            our_acc = all_results["experiments"]["binary"][model]["accuracy"]
            paper_acc = paper_results.get("binary", {}).get(model, {}).get("accuracy", our_acc)
            differences.append(our_acc - paper_acc)
        
        colors = ['#4CAF50' if d >= 0 else '#F44336' for d in differences]
        ax3.barh(models, differences, color=colors, alpha=0.8)
        ax3.axvline(x=0, color='black', linewidth=0.8)
        ax3.set_xlabel('Fark (Bizim - Makale) %')
        ax3.set_title('İkili - Makale Farkı')
    
    # 4. Fark grafiği (Multiclass)
    if "multiclass" in all_results["experiments"]:
        ax4 = fig.add_subplot(gs[1, 1])
        models = list(all_results["experiments"]["multiclass"].keys())
        differences = []
        for model in models:
            our_acc = all_results["experiments"]["multiclass"][model]["accuracy"]
            paper_acc = paper_results.get("multiclass", {}).get(model, {}).get("accuracy", our_acc)
            differences.append(our_acc - paper_acc)
        
        colors = ['#4CAF50' if d >= 0 else '#F44336' for d in differences]
        ax4.barh(models, differences, color=colors, alpha=0.8)
        ax4.axvline(x=0, color='black', linewidth=0.8)
        ax4.set_xlabel('Fark (Bizim - Makale) %')
        ax4.set_title('Çok Sınıflı - Makale Farkı')
    
    fig.suptitle('EpiGraphNet - Deney Sonuçları Özeti', fontsize=16, fontweight='bold')
    
    filename = f"{output_dir}/summary_figure.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Kaydedildi: {filename}")


def main():
    parser = argparse.ArgumentParser(description="EpiGraphNet Sonuç Görselleştirme")
    parser.add_argument(
        "--results", type=str, default="all_experiments_results.json",
        help="JSON sonuç dosyası"
    )
    parser.add_argument(
        "--output-dir", type=str, default="figures",
        help="Grafiklerin kaydedileceği klasör"
    )
    args = parser.parse_args()
    
    # Sonuçları yükle
    if not os.path.exists(args.results):
        print(f"❌ Hata: {args.results} dosyası bulunamadı!")
        print("Önce 'python run_all_experiments.py' komutunu çalıştırın.")
        return
    
    with open(args.results, 'r', encoding='utf-8') as f:
        all_results = json.load(f)
    
    print("="*60)
    print("EpiGraphNet - Sonuç Görselleştirme")
    print("="*60)
    print(f"\nSonuç dosyası: {args.results}")
    print(f"Çıktı klasörü: {args.output_dir}")
    
    # Görselleştirmeleri oluştur
    print("\n📊 Doğruluk karşılaştırması oluşturuluyor...")
    plot_accuracy_comparison(all_results, PAPER_RESULTS, args.output_dir)
    
    print("\n🔥 Metrik heatmap oluşturuluyor...")
    plot_metrics_heatmap(all_results, args.output_dir)
    
    print("\n🎯 Radar chart oluşturuluyor...")
    plot_model_radar(all_results, args.output_dir)
    
    print("\n📉 Fark grafiği oluşturuluyor...")
    plot_difference_chart(all_results, PAPER_RESULTS, args.output_dir)
    
    print("\n📈 Binary vs Multiclass karşılaştırması oluşturuluyor...")
    plot_binary_vs_multiclass(all_results, args.output_dir)
    
    print("\n📊 Standart sapma grafiği oluşturuluyor...")
    plot_std_errorbar(all_results, args.output_dir)
    
    print("\n📋 Özet figür oluşturuluyor...")
    create_summary_figure(all_results, PAPER_RESULTS, args.output_dir)
    
    print(f"\n✅ Tüm grafikler '{args.output_dir}/' klasörüne kaydedildi.")


if __name__ == "__main__":
    main()
