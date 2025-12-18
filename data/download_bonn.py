"""
Bonn Epileptik EEG Veri Kümesi İndirme Scripti

Resmi Kaynak: https://www.upf.edu/web/ntsa/downloads (UPF NTSA - Ralph Andrzejak)
Referans: Andrzejak RG, et al. (2001) Phys. Rev. E, 64, 061907

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
from typing import Optional


# Bonn veri kümesi resmi URL'leri
# Kaynak 1: Bonn Üniversitesi Epileptoloji (Orijinal)
# Kaynak 2: UPF NTSA (Yedek)
BONN_SET_URLS = {
    # Orijinal Bonn Üniversitesi URL'leri (daha güvenilir)
    "Z": "http://epileptologie-bonn.de/cms/upload/workgroup/lehnertz/Z.zip",  # Set A
    "O": "http://epileptologie-bonn.de/cms/upload/workgroup/lehnertz/O.zip",  # Set B
    "N": "http://epileptologie-bonn.de/cms/upload/workgroup/lehnertz/N.zip",  # Set C
    "F": "http://epileptologie-bonn.de/cms/upload/workgroup/lehnertz/F.zip",  # Set D
    "S": "http://epileptologie-bonn.de/cms/upload/workgroup/lehnertz/S.zip",  # Set E
}

# Yedek URL'ler (UPF NTSA)
BONN_SET_URLS_BACKUP = {
    "Z": "https://www.upf.edu/documents/229517819/234490509/Z.zip/9c4a0084-c0d6-3cf6-fe48-8a8767713e67",
    "O": "https://www.upf.edu/documents/229517819/234490509/O.zip/f324f98f-1ade-e912-b89d-e313ac362b6a",
    "N": "https://www.upf.edu/documents/229517819/234490509/N.zip/d4f08e2d-3b27-1a6a-20fe-96dcf644902b",
    "F": "https://www.upf.edu/documents/229517819/234490509/F.zip/8219dcdd-d184-0474-e0e9-1ccbba43aaee",
    "S": "https://www.upf.edu/documents/229517819/234490509/S.zip/7647d3f7-c6bb-6d72-57f7-8f12972896a6",
}


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
        # User-Agent header ekle (bazı sunucular bunu gerektiriyor)
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


def extract_zip(zip_path: str, extract_to: str) -> bool:
    """
    ZIP dosyasını çıkarır. Alt dizinlerdeki dosyaları kök dizine çıkarır.
    
    Args:
        zip_path: ZIP dosya yolu
        extract_to: Çıkarma dizini
        
    Returns:
        Başarılı mı
    """
    try:
        print(f"Çıkarılıyor: {zip_path}")
        
        extracted_count = 0
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Tüm dosyaları listele
            for item in zip_ref.namelist():
                # Sadece .txt dosyalarını al
                if item.endswith('.txt'):
                    # Dosya adını al (alt dizin yolunu görmezden gel)
                    filename = os.path.basename(item)
                    if filename:  # Boş değilse
                        # Hedef yol
                        target_path = os.path.join(extract_to, filename)
                        
                        # ZIP'ten oku ve yaz
                        with zip_ref.open(item) as source:
                            with open(target_path, 'wb') as target:
                                target.write(source.read())
                        
                        extracted_count += 1
        
        print(f"  ✓ Çıkarma tamamlandı! ({extracted_count} dosya)")
        return True
        
    except Exception as e:
        print(f"  ✗ Çıkarma hatası: {e}")
        return False


def download_bonn_dataset(
    data_dir: str = "data/bonn",
    force_download: bool = False
) -> bool:
    """
    Bonn EEG veri kümesini orijinal kaynaktan indirir.
    
    Her seti (Z, O, N, F, S) ayrı ayrı indirir ve çıkarır.
    Orijinal URL başarısız olursa yedek URL denenir.
    
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
        txt_files = list(data_path.glob("*.txt"))
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
    print("Kaynak: Bonn Üniversitesi Epileptoloji")
    print("="*60)
    
    total_files = 0
    failed_sets = []
    
    # Her seti ayrı ayrı indir
    for set_name, url in BONN_SET_URLS.items():
        print(f"\n[{set_name}] Set indiriliyor...")
        
        zip_path = data_path / f"{set_name}.zip"
        
        # Önce orijinal URL'yi dene
        success = download_file(url, str(zip_path))
        
        # Başarısız olursa yedek URL'yi dene
        if not success and set_name in BONN_SET_URLS_BACKUP:
            print(f"  → Yedek URL deneniyor...")
            success = download_file(BONN_SET_URLS_BACKUP[set_name], str(zip_path))
        
        if success:
            # ZIP'i çıkar
            if extract_zip(str(zip_path), str(data_path)):
                # ZIP'i sil
                zip_path.unlink()
                
                # İndirilen dosyaları say
                set_files = list(data_path.glob(f"{set_name}*.txt"))
                total_files += len(set_files)
                print(f"  → {len(set_files)} dosya çıkarıldı")
            else:
                failed_sets.append(set_name)
        else:
            failed_sets.append(set_name)
    
    print("\n" + "="*60)
    
    if failed_sets:
        print(f"UYARI: Bazı setler indirilemedi: {failed_sets}")
        print("Manuel indirme için: https://www.upf.edu/web/ntsa/downloads")
        return False
    
    print(f"✓ Toplam {total_files} dosya başarıyla indirildi!")
    print(f"  Konum: {data_path.absolute()}")
    
    # Veri doğrulama
    verification = verify_dataset(str(data_path))
    if verification['is_valid']:
        print("✓ Veri kümesi doğrulandı!")
    else:
        print(f"UYARI: Veri kümesi eksik - {verification}")
    
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
        return {"error": "Dizin bulunamadı"}
    
    stats = {
        'total_files': 0,
        'classes': {},
        'signal_length': None,
        'valid': True
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
        files = list(data_path.glob(f"{prefix}*.txt"))
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
    stats['is_valid'] = True
    stats['warnings'] = []
    
    if stats['total_files'] < 500:
        stats['is_valid'] = False
        stats['warnings'].append(f"Eksik dosya! Beklenen: 500, Bulunan: {stats['total_files']}")
    
    # Gerçek Bonn verisi 4097 nokta içerir (23.6s × 173.61Hz)
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
        print_dataset_info(args.data_dir)
