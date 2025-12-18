# EpiGraphNet Implementasyon Planı

## 📋 Model Mimarisi Özeti

Makaleye göre model 4 ana bileşenden oluşuyor:

1. **Veri Önişleme** - EEG verisi yükleme ve hazırlama
2. **CNN-LSTM Modülü** - Yerel ve uzun vadeli zamansal öznitelik çıkarımı
3. **KBM (Korelasyonel Bağlantı Matrisi)** - Grafik yapısı oluşturma
4. **GCN Modülü** - Grafik düzeyinde sınıflandırma

---

## 📁 Dosya Yapısı

```
codes/
├── config/
│   └── config.yaml              # Tüm hiperparametreler
├── data/
│   ├── __init__.py
│   ├── dataset.py               # Bonn EEG veri kümesi yükleme
│   └── preprocessing.py         # Veri önişleme fonksiyonları
├── models/
│   ├── __init__.py
│   ├── cnn_module.py            # 1D CNN katmanları
│   ├── lstm_module.py           # LSTM katmanları
│   ├── cnn_lstm.py              # CNN-LSTM birleşik modül
│   ├── graph_builder.py         # KBM hesaplama ve grafik oluşturma
│   ├── gcn_module.py            # GCN katmanları
│   └── epigraphnet.py           # Ana EpiGraphNet modeli
├── utils/
│   ├── __init__.py
│   ├── metrics.py               # Değerlendirme metrikleri
│   └── visualization.py         # Görselleştirme fonksiyonları
├── train.py                     # Eğitim scripti
├── evaluate.py                  # Test/değerlendirme scripti
└── requirements.txt             # Gerekli kütüphaneler
```

---

## 🔧 Uygulama Adımları

| Adım | Dosya | Açıklama |
|------|-------|----------|
| 1 | `requirements.txt` | Gerekli kütüphaneler |
| 2 | `config/config.yaml` | Hiperparametreler (Tablo I'den) |
| 3 | `data/preprocessing.py` | EEG veri önişleme |
| 4 | `data/dataset.py` | PyTorch Dataset sınıfı |
| 5 | `models/cnn_module.py` | 1D CNN blokları |
| 6 | `models/lstm_module.py` | LSTM modülü |
| 7 | `models/cnn_lstm.py` | CNN-LSTM birleşimi |
| 8 | `models/graph_builder.py` | KBM + Eşikleme (DE/BE) |
| 9 | `models/gcn_module.py` | GraphConv + Global Pooling |
| 10 | `models/epigraphnet.py` | Ana model sınıfı |
| 11 | `utils/metrics.py` | Accuracy, Recall, Precision, F1 |
| 12 | `train.py` | Eğitim döngüsü |
| 13 | `evaluate.py` | Test ve değerlendirme |

---

## 📊 Hiperparametreler (Makaledeki Tablo I'den + Şekil 1'den)

| Parametre | Değer | Açıklama |
|-----------|-------|----------|
| Batch size | 64 | Her yığındaki örnek sayısı |
| Learning rate | 0.001 | Adam optimizasyon algoritması için öğrenme hızı |
| Weight decay | 0.0005 | L2 düzenleme (regularization) parametresi |
| Epoch sayısı | 50 | Toplam eğitim epoch sayısı |
| **CNN katman sayısı** | **3** | **Şekil 1: Conv1D → MaxPool → Conv1D → Conv1D** |
| **MaxPool konumu** | **Sadece 1. katman** | **Şekil 1'de MaxPool sadece ilk Conv1D'den sonra** |
| **LSTM katman sayısı** | **2** | **Şekil 1: 2 adet LSTM Katmanı** |
| LSTM hidden size | 64 | LSTM gizli durum boyutu |
| **GCN katman sayısı** | **3** | **Şekil 1: 3 adet GraphConv Katmanı** |
| GCN hidden channels | 64 | GCN modülündeki kanal boyutu |
| Sparsity (a) | 50 | KBM eşikleme için seyreklik parametresi (0 seyrek; 100 tamamen bağlı) |
| Dropout | 0.1 | Uygulanan dropout oranı |
| LR Scheduler | LinearLR | Başlangıç faktörü 0.1 ile lineer öğrenme hızı planlayıcısı |

---

## 📐 Matematiksel Formüller

### CNN-LSTM Modülü (Eşitlik 2-5)

**Şekil 1'e göre CNN akışı:** `Conv1D → MaxPool → Conv1D → Conv1D → FC → Dropout`

```
# İLK KATMAN (MaxPool VAR):
x^(1) = BatchNorm(MaxPool(ReLU(Conv1D(x; W_1, b_1))))

# SONRAKI KATMANLAR (MaxPool YOK):
x^(l) = BatchNorm(ReLU(Conv1D(x^(l-1); W_l, b_l)))  # l = 2, 3

# FC ve Dropout:
z = FC(flatten(x^L_CNN))
z_drop = Dropout(z, p)

# LSTM (2 katman):
H_t, (h_t, c_t) = LSTM(z_drop, t=1,...,T)
```

### KBM Hesaplama (Eşitlik 6-10)
```
x̄_i^(k) = (1/T) * Σ x_i^(k)(t)           # Ortalama
x̃_i^(k)(t) = x_i^(k)(t) - x̄_i^(k)        # Merkezileştirme
V_ij^(k) = (1/(T-1)) * Σ x̃_i^(k)(t) * x̃_j^(k)(t)  # Kovaryans
σ_i^(k) = sqrt(V_ii^(k))                  # Standart sapma
C_ij^(k) = V_ij^(k) / (σ_i^(k) * σ_j^(k)) # Korelasyon
```

### Eşikleme Yöntemleri

**Değer Eşikleme (DE) - Eşitlik 11:**
```
Kenar_ij^(k) = 1, if C_ij^(k) > percentile(C^(k), 100-a)
               0, otherwise
```

**Bağlantı Eşikleme (BE) - Eşitlik 12-13:**
```
n_bağ = floor(N * a / 100)
Kenar_ij^(k) = C_ij^(k), if j ∈ S_i^(k)
               0, otherwise
```

### GCN Modülü (Eşitlik 14-16)

**Şekil 1'e göre GCN akışı:** `GraphConv → GraphConv → GraphConv → Global Max Pool → FC → Dropout → Softmax`

```
# 3 adet GraphConv katmanı:
G_i^(l+1) = ReLU(W^(l) * (G_i^(l) + Σ G_j^(l)))  # j ∈ N(i), l = 1, 2, 3

# Global Max Pooling:
g_graf = global_max_pool(G^(L), s)

# FC → Dropout → Softmax (Şekil 1'e göre):
ŷ = softmax(Dropout(FC(g_graf), p))
```

---

## 📦 Veri Kümesi Bilgileri (Bonn EEG)

- **Örnekleme frekansı:** 173.61 Hz
- **Kayıt süresi:** 23.6 saniye
- **Örnek sayısı:** 500 (her sınıftan 100)
- **Sınıflar:**
  - Sınıf 1: Nöbet esnasındaki kayıtlar
  - Sınıf 2: Nöbet geçirmeyen epilepsi hastası - nötr ortam
  - Sınıf 3: Nöbet geçirmeyen epilepsi hastası - nöbet uyaran ortam
  - Sınıf 4: Sağlıklı birey - gözler kapalı
  - Sınıf 5: Sağlıklı birey - gözler açık

- **İkili sınıflandırma:** Sınıf 1 (nöbet var) vs Sınıf 0 (nöbet yok)
- **Veri bölümü:** %80 eğitim, %10 validasyon, %10 test

---

## ✅ Kod Standartları

- PEP 8 uyumlu kod
- Type hints kullanımı
- Docstring'ler (Google style)
- Modüler yapı (Single Responsibility Principle)
- Her dosya maksimum ~150-200 satır
