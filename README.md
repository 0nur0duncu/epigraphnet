# EpiGraphNet

**Grafik Tabanlı EEG Analizi ile Epilepsi Tanı Modeli**

Bu repository, "EpiGrafNet: Grafik Temelli EEG Analizi ile Epilepsi Tanı Mimarisi" makalesinin PyTorch implementasyonunu içermektedir.

## 🏗️ Mimari

```
EEG Sinyali → [CNN-LSTM] → [KBM (Korelasyon)] → [Grafik] → [GCN] → Sınıflandırma
```

### Bileşenler

1. **CNN-LSTM Modülü**: Yerel ve uzun vadeli zamansal öznitelik çıkarımı
2. **KBM (Korelasyonel Bağlantı Matrisi)**: Öznitelikler arası korelasyon hesaplama
3. **Grafik Oluşturucu**: Değer/Bağlantı eşikleme ile grafik yapısı
4. **GCN Modülü**: Grafik konvolüsyon ve sınıflandırma

## 📁 Dosya Yapısı

```
codes/
├── config/
│   └── config.yaml          # Hiperparametreler
├── data/
│   ├── __init__.py
│   ├── dataset.py           # Bonn EEG Dataset
│   └── preprocessing.py     # Veri önişleme
├── models/
│   ├── __init__.py
│   ├── cnn_module.py        # 1D CNN
│   ├── lstm_module.py       # LSTM
│   ├── cnn_lstm.py          # CNN-LSTM birleşik
│   ├── graph_builder.py     # KBM ve grafik oluşturma
│   ├── gcn_module.py        # GCN sınıflandırıcı
│   └── epigraphnet.py       # Ana model
├── utils/
│   ├── __init__.py
│   ├── metrics.py           # Değerlendirme metrikleri
│   └── visualization.py     # Görselleştirme
├── train.py                 # Eğitim scripti
├── evaluate.py              # Değerlendirme scripti
├── demo.py                  # Demo scripti
└── requirements.txt         # Gereksinimler
```

## 🚀 Kurulum

```bash
# 1. Gereksinimleri yükle
pip install -r requirements.txt

# 2. PyTorch Geometric (GCN için - opsiyonel)
pip install torch-geometric

# 3. (Opsiyonel) CUDA desteği için
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
```

## 📊 Veri Kümesi

Bonn Epileptik EEG Veri Kümesi kullanılmaktadır:
- **Örnekleme frekansı**: 173.61 Hz
- **Kayıt süresi**: 23.6 saniye
- **Sinyal uzunluğu**: 4097 nokta
- **Toplam örnek**: 500 (her sınıftan 100)
- **5 Sınıf**:
  - S: Nöbet esnasındaki kayıtlar
  - N: Epilepsi hastası - nötr ortam
  - F: Epilepsi hastası - uyaran ortam
  - O: Sağlıklı - gözler kapalı
  - Z: Sağlıklı - gözler açık

### Veri Kümesi İndirme

```bash
# Gerçek Bonn EEG veri kümesini otomatik indir (UPF NTSA kaynağından):
python data/download_bonn.py

# Mevcut veriyi sil ve yeniden indir:
python data/download_bonn.py --force

# Veri kümesini doğrulamak için:
python data/download_bonn.py --verify-only
```

**Kaynak:** [UPF NTSA - Ralph Andrzejak Lab](https://www.upf.edu/web/ntsa/downloads)

Veri dosyaları `data/bonn/` klasörüne indirilir (Z001.txt, O001.txt, N001.txt, F001.txt, S001.txt, ... formatında).

## 🎯 Kullanım

### Demo (Gerçek veya sentetik veri ile test)
```bash
python demo.py
```

### Eğitim
```bash
python train.py --config config/config.yaml
```

### Değerlendirme
```bash
python evaluate.py --config config/config.yaml --checkpoint checkpoints/best_model.pt --num-runs 5
```

## ⚙️ Konfigürasyon

`config/config.yaml` dosyasından ayarlar değiştirilebilir:

| Parametre | Varsayılan | Açıklama |
|-----------|------------|----------|
| `batch_size` | 64 | Batch boyutu |
| `learning_rate` | 0.001 | Öğrenme hızı |
| `sparsity` | 50 | Seyreklik (a) parametresi |
| `thresholding` | "value" | Eşikleme yöntemi ("value" veya "connection") |
| `num_epochs` | 50 | Epoch sayısı |

## 📈 Sonuçlar (Makaleden)

| Model | Doğruluk | Duyarlılık | Kesinlik | F1 |
|-------|----------|------------|----------|-----|
| EpiGraphNet_DE (a=25) | **99.56%** | **99.31%** | **99.45%** | **99.36%** |
| EpiGraphNet_BE (a=25) | 99.12% | 98.15% | 99.15% | 98.63% |

## 📝 Referans

```bibtex
@article{epigraphnet2024,
  title={EpiGrafNet: Grafik Temelli EEG Analizi ile Epilepsi Tanı Mimarisi},
  author={Şimşek, Ecem and Koç, Emirhan and Koç, Aykut},
  journal={...},
  year={2024}
}
```

## 📄 Lisans

MIT License
