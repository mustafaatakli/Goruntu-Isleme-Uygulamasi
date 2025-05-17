# 📷 Görüntü İşleme Uygulaması

Bu proje, Python, OpenCV ve Tkinter kullanılarak geliştirilmiş kapsamlı bir **görüntü işleme masaüstü uygulamasıdır**. Temel görüntü işleme işlemlerinden gelişmiş filtreleme ve segmentasyona kadar birçok özelliği kullanıcı dostu bir arayüzle sunar.

---

## 🚀 Özellikler

### 📁 1. Giriş/Çıkış İşlemleri
- [x] Görsel açma (renkli ve gri tonlu)
- [x] Görsel kaydetme

### 🎨 2. Renk ve Kanal İşlemleri
- [x] Görseli griye çevirme veya gri tonda açma
- [x] R, G, B kanallarına ayırma ve gösterme
- [x] Görüntünün negatifini alma
- [x] Parlaklık artırma ve azaltma
- [x] Kontrast işlemleri
- [x] Eşikleme (Thresholding)

### 📊 3. Histogram İşlemleri
- [x] Histogram hesaplama ve görüntüleme
- [x] Histogram eşitleme

### 🔁 4. Geometrik Dönüşümler
- [x] Görüntüyü taşıma (Translation)
- [x] Aynalama (Horizontal & Vertical Flip)
- [x] Eğme (Shearing)
- [x] Döndürme (Rotate)
- [x] Kırpma (Cropping)
- [x] Perspektif düzeltme (Kullanıcı etkileşimli)

### 🧹 5. Filtreleme Teknikleri
- [x] Ortalama filtre (Mean)
- [x] Medyan filtre
- [x] Gaussian filtresi
- [x] Konservatif filtreleme
- [x] Band geçiren ve band durduran filtre
- [x] Homomorfik filtreleme

### 🧠 6. Kenar Algılama & Segmentasyon
- [x] Sobel
- [x] Prewitt
- [x] Roberts Cross
- [x] Compass
- [x] Canny
- [x] Laplace
- [x] Gabor
- [x] Hough dönüşümü
- [x] k-means segmentasyon

### ⚙️ 7. Morfolojik İşlemler
- [x] Erode
- [x] Dilate

---

## 🛠️ Kurulum

1. Python (>=3.10) yüklü olmalıdır.
2. Gerekli kütüphaneleri yüklemek için:

```bash
pip install opencv-python numpy matplotlib scipy

## Alternatif olarak, uygulamanın **.exe versiyonu** da proje dizininde yer almaktadır. Python kurulu olmayan sistemlerde bu dosya ile doğrudan çalıştırılabilir:

```bash
/GoruntuIslemeUygulamasi.exe

