"""
Bonn Epileptik EEG Veri Kümesi İndirme Scripti

Kaynak: GitHub - RYH2077/EEG-Epilepsy-Datasets
Orijinal Referans: Andrzejak RG, et al. (2001) Phys. Rev. E, 64, 061907

Veri kümesi yapısı:
- Set A (Z): 100 segment, sağlıklı gönüllüler, gözler açık
- Set B (O): 100 segment, sağlıklı gönüllüler, gözler kapalı  
- Set C (N): 100 segment, epilepsi hastaları (nöbet olmayan dönem), hipokampus
- Set D (F): 100 segment, epilepsi hastaları (nöbet olmayan dönem), epileptojenik bölge
- Set E (S): 100 segment, epilepsi hastaları, nöbet aktivitesi

Her segment: 23.6 saniye, 173.61 Hz örnekleme, 4097 veri noktası (ASCII formatında)
"""

import os
import zipfile
import urllib.request
from pathlib import Path


# Tam Bonn veri seti (500 dosya)
# Kaynak: RYH2077/EEG-Epilepsy-Datasets
DATASET_URL = "https://github.com/RYH2077/EEG-Epilepsy-Datasets/raw/master/Bonn%20EEG%20dataset.zip"


def download_file(url: str, save_path: str, chunk_size: int = 8192) -> bool:
    """
    URL'den dosya indirir.
    
    Args:
        url: İndirme URL'si
        save_path: Kayıt yolu
        chunk_size: İndirme parça boyutu
        
    Returns:
        Başarılı mı
    """
    try:
        req = urllib.request.Request(
            url,
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        )
        
        with urllib.request.urlopen(req, timeout=120) as response:
            total_size = int(response.headers.get('Content-Length', 0))
            downloaded = 0
            
            with open(save_path, 'wb') as f:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        print(f"\r  İlerleme: {percent:.1f}%", end="", flush=True)
        
        print("\n  ✓ İndirme tamamlandı!")
        return True
        
    except Exception as e:
        print(f"\n  ✗ İndirme hatası: {e}")
        return False


def download_bonn_dataset(
    data_dir: str = "data/bonn",
    force_download: bool = False
) -> bool:
    """
    Bonn EEG veri kümesini GitHub'dan indirir.
    
    Args:
        data_dir: Hedef dizin
        force_download: Mevcut veriyi sil ve yeniden indir
        
    Returns:
        Başarılı mı
    """
    import shutil
    
    data_path = Path(data_dir)
    
    # Mevcut veri kontrolü
    if data_path.exists() and not force_download:
        txt_files = list(data_path.glob("*.txt")) + list(data_path.glob("*.TXT"))
        if len(txt_files) >= 500:
            print(f"✓ Veri zaten mevcut: {data_dir} ({len(txt_files)} dosya)")
            return True
    
    # Force download ise mevcut veriyi sil
    if force_download and data_path.exists():
        print(f"Mevcut veri siliniyor: {data_dir}")
        shutil.rmtree(data_path)
    
    # Dizin oluştur
    data_path.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("Bonn Epileptik EEG Veri Kümesi İndirme")
    print("Kaynak: GitHub - RYH2077/EEG-Epilepsy-Datasets")
    print("="*60)
    
    # ZIP dosyasını indir
    zip_path = data_path / "bonn_dataset.zip"
    
    print("\nVeri seti indiriliyor...")
    success = download_file(DATASET_URL, str(zip_path))
    
    if not success:
        print("✗ İndirme başarısız!")
        return False
    
    # ZIP'i çıkar
    try:
        print(f"\nÇıkarılıyor: {zip_path}")
        
        extracted_count = 0
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            for item in zip_ref.namelist():
                # .txt dosyalarını bul (case-insensitive)
                if item.lower().endswith('.txt'):
                    filename = os.path.basename(item)
                    if filename:
                        target_path = os.path.join(data_dir, filename)
                        with zip_ref.open(item) as source:
                            with open(target_path, 'wb') as target:
                                target.write(source.read())
                        extracted_count += 1
        
        print(f"  ✓ {extracted_count} dosya çıkarıldı!")
        
        # ZIP'i sil
        zip_path.unlink()
        
    except Exception as e:
        print(f"  ✗ Çıkarma hatası: {e}")
        return False
    
    print("\n" + "="*60)
    print(f"✓ Toplam {extracted_count} dosya başarıyla indirildi!")
    print(f"  Konum: {data_path.absolute()}")
    
    # Veri doğrulama
    verification = verify_dataset(str(data_path))
    if verification['is_valid']:
        print("✓ Veri kümesi doğrulandı!")
    else:
        for warning in verification.get('warnings', []):
            print(f"UYARI: {warning}")
    
    return True


def verify_dataset(data_dir: str = "data/bonn") -> dict:
    """
    Veri kümesini doğrular ve istatistikleri döndürür.
    
    Args:
        data_dir: Veri dizini
        
    Returns:
        Veri kümesi istatistikleri
    """
    import numpy as np
    
    data_path = Path(data_dir)
    
    if not data_path.exists():
        return {"error": "Dizin bulunamadı", "is_valid": False}
    
    stats = {
        'total_files': 0,
        'classes': {},
        'signal_length': None,
        'is_valid': True,
        'warnings': []
    }
    
    prefixes = ['Z', 'O', 'N', 'F', 'S']
    class_names = {
        'Z': 'Sağlıklı (gözler açık)',
        'O': 'Sağlıklı (gözler kapalı)',
        'N': 'Epilepsi (nötr)',
        'F': 'Epilepsi (uyaran)',
        'S': 'Nöbet'
    }
    
    for prefix in prefixes:
        # Hem küçük hem büyük harf uzantıları ara
        files = list(data_path.glob(f"{prefix}*.txt")) + list(data_path.glob(f"{prefix}*.TXT"))
        stats['classes'][prefix] = {
            'count': len(files),
            'name': class_names[prefix]
        }
        stats['total_files'] += len(files)
        
        # İlk dosyayı kontrol et
        if files and stats['signal_length'] is None:
            signal = np.loadtxt(files[0])
            stats['signal_length'] = len(signal)
    
    # Doğrulama
    if stats['total_files'] < 500:
        stats['is_valid'] = False
        stats['warnings'].append(f"Eksik dosya! Beklenen: 500, Bulunan: {stats['total_files']}")
    
    if stats['signal_length'] is not None and stats['signal_length'] != 4097:
        stats['warnings'].append(f"Sinyal uzunluğu: {stats['signal_length']} (Beklenen: 4097)")
    
    return stats


def print_dataset_info(data_dir: str = "data/bonn") -> None:
    """Veri kümesi bilgilerini yazdırır."""
    stats = verify_dataset(data_dir)
    
    print("\n" + "="*50)
    print("BONN EEG VERİ KÜMESİ BİLGİLERİ")
    print("="*50)
    
    if 'error' in stats:
        print(f"HATA: {stats['error']}")
        return
    
    print(f"Toplam dosya: {stats['total_files']}")
    print(f"Sinyal uzunluğu: {stats['signal_length']} nokta")
    print(f"Örnekleme frekansı: 173.61 Hz")
    print(f"Kayıt süresi: 23.6 saniye")
    print()
    
    print("Sınıf Dağılımı:")
    print("-"*40)
    for prefix, info in stats['classes'].items():
        print(f"  {prefix}: {info['count']:3d} dosya - {info['name']}")
    
    print()
    if stats['is_valid']:
        print("✓ Veri kümesi geçerli!")
    else:
        for warning in stats.get('warnings', []):
            print(f"✗ UYARI: {warning}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Bonn EEG Veri Kümesi İndirme")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/bonn",
        help="Veri dizini"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Mevcut veriyi sil ve yeniden indir"
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Sadece mevcut veriyi doğrula"
    )
    args = parser.parse_args()
    
    if args.verify_only:
        print_dataset_info(args.data_dir)
    else:
        download_bonn_dataset(args.data_dir, args.force)
