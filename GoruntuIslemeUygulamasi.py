import tkinter as tk
from tkinter import ttk, filedialog, Scale, HORIZONTAL, Scrollbar, simpledialog, messagebox
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image, ImageTk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import traceback

try:
    from islemler import erode_islem, dilate_islem, opening_islem, closing_islem, kernel_goster, \
        morfolojik_islev_karsilastir

except ImportError as e:
    print(f"modul içe aktarılamadi: {e}")

try:
    print("modul import ediliyor...")
    from ozellikler import hough_dogru_bulma, hough_cember_bulma, kmeans_segmentation

except ImportError as e:
    print(f"modul ice aktarilamadi: {e}")

def define_fallback_functions():
    global erode_islem, dilate_islem, opening_islem, closing_islem, kernel_goster
    global hough_dogru_bulma, hough_cember_bulma, kmeans_segmentation, gabor_filtre, morfolojik_islev_karsilastir

    if 'erode_islem' not in globals():
        def erode_islem(img, kernel_size=3, kernel_shape="kare", iterations=1):
            try:
                if img is None:
                    return None

                if kernel_shape.lower() == "kare":
                    kernel = np.ones((kernel_size, kernel_size), np.uint8)
                elif kernel_shape.lower() == "disk":
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
                elif kernel_shape.lower() == "çapraz" or kernel_shape.lower() == "capraz":
                    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (kernel_size, kernel_size))
                elif kernel_shape.lower() == "elips":
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
                else:
                    kernel = np.ones((kernel_size, kernel_size), np.uint8)

                if len(img.shape) > 2:
                    b, g, r = cv2.split(img)
                    b_eroded = cv2.erode(b, kernel, iterations=iterations)
                    g_eroded = cv2.erode(g, kernel, iterations=iterations)
                    r_eroded = cv2.erode(r, kernel, iterations=iterations)
                    processed_image = cv2.merge([b_eroded, g_eroded, r_eroded])
                else:
                    processed_image = cv2.erode(img, kernel, iterations=iterations)

                return processed_image
            except Exception as e:
                print(f"Yedek erode_islem fonksiyonunda hata: {str(e)}")
                traceback.print_exc()
                return None

    if 'hough_dogru_bulma' not in globals():
        def hough_dogru_bulma(img, rho=1, theta=np.pi / 180, esik=150, min_cizgi_uzunlugu=None, max_bosluk=None):
            try:
                if img is None:
                    return None, None, None

                output = img.copy()

                if len(img.shape) > 2:
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                else:
                    gray = img.copy()
                    output = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

                edges = cv2.Canny(gray, 50, 150, apertureSize=3)

                if min_cizgi_uzunlugu is not None and max_bosluk is not None:
                    lines = cv2.HoughLinesP(edges, rho, theta, esik, minLineLength=min_cizgi_uzunlugu,
                                            maxLineGap=max_bosluk)

                    if lines is not None:
                        for line in lines:
                            x1, y1, x2, y2 = line[0]
                            cv2.line(output, (x1, y1), (x2, y2), (0, 255, 0), 2)
                else:
                    lines = cv2.HoughLines(edges, rho, theta, esik)

                    if lines is not None:
                        for line in lines:
                            rho_val, theta_val = line[0]
                            a = np.cos(theta_val)
                            b = np.sin(theta_val)
                            x0 = a * rho_val
                            y0 = b * rho_val
                            x1 = int(x0 + 1000 * (-b))
                            y1 = int(y0 + 1000 * (a))
                            x2 = int(x0 - 1000 * (-b))
                            y2 = int(y0 - 1000 * (a))
                            cv2.line(output, (x1, y1), (x2, y2), (0, 255, 0), 2)

                return gray, edges, output
            except Exception as e:
                print(f"Yedek hough_dogru_bulma fonksiyonunda hata: {str(e)}")
                traceback.print_exc()
                return None, None, None

    if 'hough_cember_bulma' not in globals():
        def hough_cember_bulma(img, dp=1.0, min_dist=30, param1=50, param2=30, min_radius=10, max_radius=100):
            try:
                if img is None:
                    return None, None

                output = img.copy()
                if len(img.shape) > 2:
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                else:
                    gray = img.copy()
                    output = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

                blurred = cv2.GaussianBlur(gray, (9, 9), 2)

                circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp, min_dist,
                                           param1=param1, param2=param2,
                                           minRadius=min_radius, maxRadius=max_radius)

                if circles is not None:
                    circles = np.uint16(np.around(circles))
                    for i in circles[0, :]:

                        cv2.circle(output, (i[0], i[1]), i[2], (0, 255, 0), 2)

                        cv2.circle(output, (i[0], i[1]), 2, (0, 0, 255), 3)

                return gray, output
            except Exception as e:
                print(f"Yedek hough_cember_bulma fonksiyonunda hata: {str(e)}")
                traceback.print_exc()
                return None, None

    if 'gabor_filtre' not in globals():
        def gabor_filtre(img, kernel_size=21, sigma=5, theta=np.pi / 4, lambd=10, gamma=0.5, psi=0, normalize=True):
            try:

                if len(img.shape) > 2:
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                else:
                    gray = img.copy()
                kernel = cv2.getGaborKernel(
                    (kernel_size, kernel_size),
                    sigma,
                    theta,
                    lambd,
                    gamma,
                    psi,
                    ktype=cv2.CV_32F
                )

                kernel /= np.sum(np.abs(kernel))
                filtered_img = cv2.filter2D(gray, cv2.CV_8UC3, kernel)

                if normalize:
                    filtered_img = cv2.normalize(filtered_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

                return filtered_img

            except Exception as e:
                print(f"Yedek gabor_filtre fonksiyonunda hata: {str(e)}")
                traceback.print_exc()
                return None

    if 'kmeans_segmentation' not in globals():
        def kmeans_segmentation(img, k=3, attempts=10):
            try:
                if img is None:
                    print("Hata: İşlenecek görüntü bulunamadı!")
                    return None, None, None

                pixel_values = img.reshape((-1, 3))
                pixel_values = np.float32(pixel_values)
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
                _, labels, centers = cv2.kmeans(pixel_values, k, None, criteria, attempts, cv2.KMEANS_RANDOM_CENTERS)
                centers = np.uint8(centers)
                segmented_data = centers[labels.flatten()]
                segmented_image = segmented_data.reshape(img.shape)
                label_image = labels.reshape(img.shape[0], img.shape[1])
                return img, segmented_image, label_image

            except Exception as e:
                print(f"Yedek kmeans_segmentation fonksiyonunda hata: {str(e)}")
                import traceback
                traceback.print_exc()
                return None, None, None

define_fallback_functions()

class GoruntuIslemeUygulamasi:
    def __init__(self, root):
        self.root = root
        self.root.title("Görüntü İşleme Uygulaması")
        self.root.geometry("1200x700")

        self.drawing_mode = None
        self.start_point = None
        self.temp_image = None

        self.cizim_rengi_varsayilan = "red"
        self.kalinlik_varsayilan = 2

        self.cizim_rengi = tk.StringVar(value=self.cizim_rengi_varsayilan)
        self.kalinlik_scale_value = tk.IntVar(value=self.kalinlik_varsayilan)

        self.menubar = tk.Menu(self.root)
        self.root.config(menu=self.menubar)

        self.dosya_menu = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label="Dosya", menu=self.dosya_menu)
        self.dosya_menu.add_command(label="Aç", command=self.goruntu_yukle)
        self.dosya_menu.add_command(label="Kaydet", command=self.goruntu_kaydet)
        self.dosya_menu.add_command(label="Orijinal Görüntüye Dön", command=self.orijinal_goruntu_don)
        self.dosya_menu.add_separator()
        self.dosya_menu.add_command(label="Çıkış", command=self.root.quit)

        self.duzenleme_menu = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label="Düzenleme", menu=self.duzenleme_menu)
        self.duzenleme_menu.add_command(label="Parlaklık Ayarla", command=self.parlaklik_uygula)
        self.duzenleme_menu.add_command(label="Kontrast Ayarla", command=self.kontrast_uygula)
        self.duzenleme_menu.add_command(label="Kırpma", command=self.kirpma_uygula)
        self.duzenleme_menu.add_separator()
        self.duzenleme_menu.add_command(label="Kontrast Germe", command=self.kontrast_germe_uygula)

        self.goruntu_menu = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label="Görüntü", menu=self.goruntu_menu)
        self.goruntu_menu.add_command(label="Griye Çevir", command=self.griye_cevir_uygula)
        self.goruntu_menu.add_command(label="Negatife Çevir", command=self.negatife_cevir_uygula)


        self.kanallar_menu = tk.Menu(self.goruntu_menu, tearoff=0)
        self.goruntu_menu.add_cascade(label="Kanalları Göster", menu=self.kanallar_menu)
        self.kanallar_menu.add_command(label="R Kanalı", command=lambda: self.kanal_goster_uygula('red'))
        self.kanallar_menu.add_command(label="G Kanalı", command=lambda: self.kanal_goster_uygula('green'))
        self.kanallar_menu.add_command(label="B Kanalı", command=lambda: self.kanal_goster_uygula('blue'))

        self.goruntu_menu.add_separator()
        self.goruntu_menu.add_command(label="Histogram Göster", command=self.histogram_goster)
        self.goruntu_menu.add_command(label="Histogram Eşitleme", command=self.histogram_esitle)

        self.sekil_cizme_menu = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label="Şekil Çizme", menu=self.sekil_cizme_menu)

        self.sekil_cizme_menu.add_command(label="Çizgi Çiz", command=self.cizgi_ciz_uygula)
        self.sekil_cizme_menu.add_command(label="Dikdörtgen Çiz", command=self.dikdortgen_ciz_uygula)
        self.sekil_cizme_menu.add_command(label="Daire Çiz", command=self.daire_ciz_uygula)

        self.filtreler_menu = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label="Filtreler", menu=self.filtreler_menu)

        self.yumusatma_menu = tk.Menu(self.filtreler_menu, tearoff=0)
        self.filtreler_menu.add_cascade(label="Yumuşatma Filtreleri", menu=self.yumusatma_menu)
        self.yumusatma_menu.add_command(label="Ortalama Filtre", command=self.ortalama_filtre_uygula)
        self.yumusatma_menu.add_command(label="Gauss Filtresi", command=self.gauss_filtre_uygula)
        self.yumusatma_menu.add_command(label="Medyan Filtresi", command=self.medyan_filtre_uygula)

        self.frekans_menu = tk.Menu(self.filtreler_menu, tearoff=0)
        self.filtreler_menu.add_cascade(label="Frekans Filtreleri", menu=self.frekans_menu)
        self.frekans_menu.add_command(label="Alçak Geçiren", command=self.alcak_geciren_filtre_uygula)
        self.frekans_menu.add_command(label="Yüksek Geçiren", command=self.yuksek_geciren_filtre_uygula)
        self.frekans_menu.add_command(label="Gauss Alçak Geçiren", command=self.gauss_alcak_geciren_uygula)
        self.frekans_menu.add_command(label="Gauss Yüksek Geçiren", command=self.gauss_yuksek_geciren_uygula)

        self.kenar_menu = tk.Menu(self.filtreler_menu, tearoff=0)
        self.filtreler_menu.add_cascade(label="Kenar Bulma", menu=self.kenar_menu)
        self.kenar_menu.add_command(label="Sobel", command=lambda: self.kenar_bulma_uygula('sobel'))
        self.kenar_menu.add_command(label="Prewitt", command=lambda: self.kenar_bulma_uygula('prewitt'))
        self.kenar_menu.add_command(label="Roberts Cross", command=lambda: self.kenar_bulma_uygula('roberts'))
        self.kenar_menu.add_command(label="Compass", command=lambda: self.kenar_bulma_uygula('compass'))
        self.kenar_menu.add_command(label="Canny", command=lambda: self.kenar_bulma_uygula('canny'))
        self.kenar_menu.add_command(label="Laplace", command=lambda: self.kenar_bulma_uygula('laplace'))

        self.morfolojik_menu = tk.Menu(self.filtreler_menu, tearoff=0)
        self.filtreler_menu.add_cascade(label="Morfolojik İşlemler", menu=self.morfolojik_menu)
        self.morfolojik_menu.add_command(label="Aşındırma (Erode)", command=self.erode_uygula)
        self.morfolojik_menu.add_command(label="Genişletme (Dilate)", command=self.dilate_uygula)
        self.morfolojik_menu.add_command(label="Açma (Opening)", command=self.opening_uygula)
        self.morfolojik_menu.add_command(label="Kapama (Closing)", command=self.closing_uygula)
        self.morfolojik_menu.add_command(label="Yapısal Elemanları Göster", command=self.kernel_goster_uygula)

        self.filtreler_menu.add_separator()
        self.filtreler_menu.add_command(label="Gabor Filtresi", command=self.gabor_filtre_uygula)

        self.donusumler_menu = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label="Dönüşümler", menu=self.donusumler_menu)
        self.donusumler_menu.add_command(label="Döndürme", command=self.dondurme_uygula)
        self.donusumler_menu.add_command(label="Ölçekleme", command=self.olcekleme_uygula)
        self.donusumler_menu.add_command(label="Taşıma", command=self.tasima_uygula)

        self.egme_menu = tk.Menu(self.donusumler_menu, tearoff=0)
        self.donusumler_menu.add_cascade(label="Eğme", menu=self.egme_menu)
        self.egme_menu.add_command(label="X Yönünde Eğme", command=self.egme_x_uygula)
        self.egme_menu.add_command(label="Y Yönünde Eğme", command=self.egme_y_uygula)

        self.aynalama_menu = tk.Menu(self.donusumler_menu, tearoff=0)
        self.donusumler_menu.add_cascade(label="Aynalama", menu=self.aynalama_menu)
        self.aynalama_menu.add_command(label="Dikey Aynalama", command=self.aynalama_dikey_uygula)
        self.aynalama_menu.add_command(label="Yatay Aynalama", command=self.aynalama_yatay_uygula)
        self.aynalama_menu.add_command(label="Dikey ve Yatay Aynalama", command=self.aynalama_her_iki_uygula)

        self.donusumler_menu.add_separator()
        self.donusumler_menu.add_command(label="Perspektif Düzeltme", command=self.perspektif_duzeltme_uygula)

        self.esikleme_menu = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label="Eşikleme", menu=self.esikleme_menu)
        self.esikleme_menu.add_command(label="İkili Eşikleme", command=self.ikili_esikleme_uygula)
        self.esikleme_menu.add_command(label="Otsu Eşikleme", command=self.otsu_esikleme_uygula)
        self.esikleme_menu.add_command(label="Adaptif Eşikleme", command=self.adaptif_esikleme_uygula)

        self.ileri_duzey_menu = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label="İleri Düzey", menu=self.ileri_duzey_menu)
        self.ileri_duzey_menu.add_command(label="Hough Doğru Tespiti", command=self.hough_dogru_uygula)
        self.ileri_duzey_menu.add_command(label="Hough Çember Tespiti", command=self.hough_cember_uygula)
        self.ileri_duzey_menu.add_command(label="K-Means Segmentasyon", command=self.kmeans_uygula)

        self.main_frame = tk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        self.sol_panel = tk.Frame(self.main_frame, width=280, bg="#f0f0f0", relief=tk.RAISED, borderwidth=1)
        self.sol_panel.pack(side=tk.LEFT, fill=tk.Y)
        self.sol_panel.pack_propagate(False)

        self.sol_panel_icerik = tk.Frame(self.sol_panel, bg="#f0f0f0")
        self.sol_panel_icerik.pack(fill=tk.BOTH, expand=True)

        self.sol_scrollbar = Scrollbar(self.sol_panel, orient="vertical")
        self.sol_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.sol_canvas = tk.Canvas(self.sol_panel_icerik, bg="#f0f0f0", yscrollcommand=self.sol_scrollbar.set,
                                    height=250)
        self.sol_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.sol_scrollbar.config(command=self.sol_canvas.yview)

        self.sol_icerik_frame = tk.Frame(self.sol_canvas, bg="#f0f0f0")


        self.sol_canvas_window = self.sol_canvas.create_window(
            (0, 0), window=self.sol_icerik_frame, anchor="nw", width=265
        )

        tk.Label(self.sol_icerik_frame, text="İŞLEMLER", font=("Arial", 14, "bold"), bg="#f0f0f0").pack(pady=10)

        self.yukle_button = tk.Button(
            self.sol_icerik_frame,
            text="Görüntü Yükle",
            bg="#4CAF50",
            fg="white",
            font=("Arial", 12),
            height=2,
            command=self.goruntu_yukle
        )
        self.yukle_button.pack(fill=tk.X, padx=10, pady=5)

        self.orijinal_button = tk.Button(
            self.sol_icerik_frame,
            text="Orijinal Görüntüye Dön",
            bg="#FF9800",
            fg="white",
            font=("Arial", 10),
            height=1,
            command=self.orijinal_goruntu_don
        )
        self.orijinal_button.pack(fill=tk.X, padx=10, pady=5)

        self.goruntu_islemleri_frame = tk.LabelFrame(self.sol_icerik_frame, text="Görüntü İşlemleri", bg="#f0f0f0")
        self.goruntu_islemleri_frame.pack(fill=tk.X, padx=10, pady=5)

        self.griye_cevir_btn = tk.Button(self.goruntu_islemleri_frame, text="Griye Çevir", bg="#2196F3", fg="white",
                                         height=1, command=self.griye_cevir_uygula)
        self.griye_cevir_btn.pack(fill=tk.X, padx=5, pady=2)

        self.negatife_cevir_btn = tk.Button(self.goruntu_islemleri_frame, text="Negatife Çevir", bg="#2196F3",
                                            fg="white", height=1, command=self.negatife_cevir_uygula)
        self.negatife_cevir_btn.pack(fill=tk.X, padx=5, pady=2)

        self.dondurme_btn = tk.Button(self.goruntu_islemleri_frame, text="Döndürme", bg="#2196F3", fg="white", height=1,
                                      command=self.dondurme_uygula)
        self.dondurme_btn.pack(fill=tk.X, padx=5, pady=2)

        self.aynalama_btn = tk.Button(self.goruntu_islemleri_frame, text="Aynalama", bg="#2196F3", fg="white", height=1,
                                      command=self.aynalama_popup)
        self.aynalama_btn.pack(fill=tk.X, padx=5, pady=2)

        self.kanal_btn = tk.Button(self.goruntu_islemleri_frame, text="B Kanalı Göster", bg="#2196F3", fg="white",
                                   height=1, command=lambda: self.kanal_goster_uygula('blue'))
        self.kanal_btn.pack(fill=tk.X, padx=5, pady=2)

        self.kanal_btn = tk.Button(self.goruntu_islemleri_frame, text="G Kanalı Göster", bg="#2196F3", fg="white",
                                   height=1, command=lambda: self.kanal_goster_uygula('green'))
        self.kanal_btn.pack(fill=tk.X, padx=5, pady=2)

        self.kanal_btn = tk.Button(self.goruntu_islemleri_frame, text="R Kanalı Göster", bg="#2196F3", fg="white",
                                   height=1, command=lambda: self.kanal_goster_uygula('red'))
        self.kanal_btn.pack(fill=tk.X, padx=5, pady=2)

        self.donusum_islemleri_frame = tk.LabelFrame(self.sol_icerik_frame, text="Dönüşüm İşlemleri", bg="#f0f0f0")
        self.donusum_islemleri_frame.pack(fill=tk.X, padx=10, pady=5)

        tk.Label(self.donusum_islemleri_frame, text="Taşıma (X, Y):", bg="#f0f0f0").pack(anchor='w', padx=5)

        self.tasima_frame = tk.Frame(self.donusum_islemleri_frame, bg="#f0f0f0")
        self.tasima_frame.pack(fill=tk.X, padx=5, pady=2)

        tk.Label(self.tasima_frame, text="X:", bg="#f0f0f0").pack(side=tk.LEFT)
        self.tasima_x_entry = tk.Entry(self.tasima_frame, width=5)
        self.tasima_x_entry.pack(side=tk.LEFT, padx=5)
        self.tasima_x_entry.insert(0, "0")

        tk.Label(self.tasima_frame, text="Y:", bg="#f0f0f0").pack(side=tk.LEFT)
        self.tasima_y_entry = tk.Entry(self.tasima_frame, width=5)
        self.tasima_y_entry.pack(side=tk.LEFT, padx=5)
        self.tasima_y_entry.insert(0, "0")

        self.tasima_btn = tk.Button(self.donusum_islemleri_frame, text="Taşıma Uygula", bg="#2196F3", fg="white",
                                    height=1, command=self.tasima_uygula)
        self.tasima_btn.pack(fill=tk.X, padx=5, pady=2)

        tk.Label(self.donusum_islemleri_frame, text="Döndürme (Derece):", bg="#f0f0f0").pack(anchor='w', padx=5)

        self.dondurme_frame = tk.Frame(self.donusum_islemleri_frame, bg="#f0f0f0")
        self.dondurme_frame.pack(fill=tk.X, padx=5, pady=2)

        self.dondurme_deger_label = tk.Label(self.dondurme_frame, text="0°", width=4, bg="#f0f0f0")
        self.dondurme_deger_label.pack(side=tk.LEFT)

        self.dondurme_scale = Scale(
            self.dondurme_frame,
            from_=-180,
            to=180,
            orient=HORIZONTAL,
            bg="#f0f0f0",
            command=self.dondurme_degeri_degistir
        )
        self.dondurme_scale.set(0)
        self.dondurme_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.dondurme_btn = tk.Button(
            self.donusum_islemleri_frame,
            text="Döndürme Uygula",
            bg="#2196F3",
            fg="white",
            height=1,
            command=self.dondurme_uygula
        )
        self.dondurme_btn.pack(fill=tk.X, padx=5, pady=2)

        tk.Label(self.donusum_islemleri_frame, text="Ölçekleme:", bg="#f0f0f0").pack(anchor='w', padx=5)

        self.olcekleme_frame = tk.Frame(self.donusum_islemleri_frame, bg="#f0f0f0")
        self.olcekleme_frame.pack(fill=tk.X, padx=5, pady=2)

        self.olcekleme_deger_label = tk.Label(self.olcekleme_frame, text="1.0", width=3, bg="#f0f0f0")
        self.olcekleme_deger_label.pack(side=tk.LEFT)

        self.olcekleme_scale = Scale(
            self.olcekleme_frame,
            from_=0.1,
            to=3.0,
            resolution=0.1,
            orient=HORIZONTAL,
            bg="#f0f0f0",
            command=self.olcekleme_degeri_degistir
        )
        self.olcekleme_scale.set(1.0)
        self.olcekleme_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.olcekleme_btn = tk.Button(
            self.donusum_islemleri_frame,
            text="Ölçekleme Uygula",
            bg="#2196F3",
            fg="white",
            height=1,
            command=self.olcekleme_uygula
        )
        self.olcekleme_btn.pack(fill=tk.X, padx=5, pady=2)

        self.shearing_x_btn = tk.Button(
            self.donusum_islemleri_frame,
            text="X Ekseninde Eğme",
            bg="#2196F3",
            fg="white",
            height=1,
            command=self.egme_x_uygula
        )
        self.shearing_x_btn.pack(fill=tk.X, padx=5, pady=2)

        self.shearing_y_btn = tk.Button(
            self.donusum_islemleri_frame,
            text="Y Ekseninde Eğme",
            bg="#2196F3",
            fg="white",
            height=1,
            command=self.egme_y_uygula
        )
        self.shearing_y_btn.pack(fill=tk.X, padx=5, pady=2)

        self.aynalama_islemleri_frame = tk.LabelFrame(self.sol_icerik_frame, text="Aynalama İşlemleri", bg="#f0f0f0")
        self.aynalama_islemleri_frame.pack(fill=tk.X, padx=10, pady=5)

        self.aynalama_dikey_btn = tk.Button(
            self.aynalama_islemleri_frame,
            text="Dikey Eksende Aynalama",
            bg="#2196F3",
            fg="white",
            height=1,
            command=self.aynalama_dikey_uygula
        )
        self.aynalama_dikey_btn.pack(fill=tk.X, padx=5, pady=2)

        self.aynalama_yatay_btn = tk.Button(
            self.aynalama_islemleri_frame,
            text="Yatay Eksende Aynalama",
            bg="#2196F3",
            fg="white",
            height=1,
            command=self.aynalama_yatay_uygula
        )
        self.aynalama_yatay_btn.pack(fill=tk.X, padx=5, pady=2)

        self.aynalama_her_iki_btn = tk.Button(
            self.aynalama_islemleri_frame,
            text="Her İki Eksende Aynalama",
            bg="#2196F3",
            fg="white",
            height=1,
            command=self.aynalama_her_iki_uygula
        )
        self.aynalama_her_iki_btn.pack(fill=tk.X, padx=5, pady=2)

        self.kirpma_btn = tk.Button(
            self.aynalama_islemleri_frame,
            text="Görüntüyü Kırp",
            bg="#2196F3",
            fg="white",
            height=1,
            command=self.kirpma_uygula
        )
        self.kirpma_btn.pack(fill=tk.X, padx=5, pady=2)

        self.sekil_cizme_frame = tk.LabelFrame(self.sol_icerik_frame, text="Şekil Çizme İşlemleri", bg="#f0f0f0")
        self.sekil_cizme_frame.pack(fill=tk.X, padx=10, pady=5)

        self.cizgi_ciz_btn = tk.Button(
            self.sekil_cizme_frame,
            text="Çizgi Çiz",
            bg="#2196F3",
            fg="white",
            height=1,
            command=self.cizgi_ciz_uygula
        )
        self.cizgi_ciz_btn.pack(fill=tk.X, padx=5, pady=2)

        self.dikdortgen_ciz_btn = tk.Button(
            self.sekil_cizme_frame,
            text="Dikdörtgen Çiz",
            bg="#2196F3",
            fg="white",
            height=1,
            command=self.dikdortgen_ciz_uygula
        )
        self.dikdortgen_ciz_btn.pack(fill=tk.X, padx=5, pady=2)

        self.daire_ciz_btn = tk.Button(
            self.sekil_cizme_frame,
            text="Daire Çiz",
            bg="#2196F3",
            fg="white",
            height=1,
            command=self.daire_ciz_uygula
        )
        self.daire_ciz_btn.pack(fill=tk.X, padx=5, pady=2)


        tk.Label(self.sekil_cizme_frame, text="Çizim Rengi:", bg="#f0f0f0").pack(anchor='w', padx=5)
        self.renk_secim_frame = tk.Frame(self.sekil_cizme_frame, bg="#f0f0f0")
        self.renk_secim_frame.pack(fill=tk.X, padx=5, pady=2)

        self.cizim_rengi = tk.StringVar(value="red")
        self.kirmizi_rb = tk.Radiobutton(self.renk_secim_frame, text="Kırmızı", variable=self.cizim_rengi, value="red",
                                         bg="#f0f0f0")
        self.kirmizi_rb.pack(side=tk.LEFT)
        self.yesil_rb = tk.Radiobutton(self.renk_secim_frame, text="Yeşil", variable=self.cizim_rengi, value="green",
                                       bg="#f0f0f0")
        self.yesil_rb.pack(side=tk.LEFT)
        self.mavi_rb = tk.Radiobutton(self.renk_secim_frame, text="Mavi", variable=self.cizim_rengi, value="blue",
                                      bg="#f0f0f0")
        self.mavi_rb.pack(side=tk.LEFT)

        tk.Label(self.sekil_cizme_frame, text="Çizgi Kalınlığı:", bg="#f0f0f0").pack(anchor='w', padx=5)
        self.kalinlik_frame = tk.Frame(self.sekil_cizme_frame, bg="#f0f0f0")
        self.kalinlik_frame.pack(fill=tk.X, padx=5, pady=2)

        self.kalinlik_deger_label = tk.Label(self.kalinlik_frame, text="2", width=2, bg="#f0f0f0")
        self.kalinlik_deger_label.pack(side=tk.LEFT)

        self.kalinlik_scale = Scale(
            self.kalinlik_frame,
            from_=1,
            to=10,
            orient=HORIZONTAL,
            bg="#f0f0f0",
            command=self.kalinlik_degistir,
            variable=self.kalinlik_scale_value
        )
        self.kalinlik_scale.set(2)
        self.kalinlik_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)


        self.aydinlatma_islemleri_frame = tk.LabelFrame(self.sol_icerik_frame, text="Aydınlatma İşlemleri",
                                                        bg="#f0f0f0")
        self.aydinlatma_islemleri_frame.pack(fill=tk.X, padx=10, pady=5)

        tk.Label(self.aydinlatma_islemleri_frame, text="Parlaklık Değeri:", bg="#f0f0f0").pack(anchor='w', padx=5)

        self.parlaklik_scale_frame = tk.Frame(self.aydinlatma_islemleri_frame, bg="#f0f0f0")
        self.parlaklik_scale_frame.pack(fill=tk.X, padx=5, pady=2)

        self.parlaklik_deger_label = tk.Label(self.parlaklik_scale_frame, text="0", width=3, bg="#f0f0f0")
        self.parlaklik_deger_label.pack(side=tk.LEFT)

        self.parlaklik_scale = Scale(
            self.parlaklik_scale_frame,
            from_=-100,
            to=100,
            orient=HORIZONTAL,
            bg="#f0f0f0",
            command=self.parlaklik_degistir
        )
        self.parlaklik_scale.set(0)
        self.parlaklik_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.parlaklik_uygula_btn = tk.Button(
            self.aydinlatma_islemleri_frame,
            text="Parlaklık Uygula",
            bg="#2196F3",
            fg="white",
            height=1,
            command=self.parlaklik_uygula
        )
        self.parlaklik_uygula_btn.pack(fill=tk.X, padx=5, pady=2)

        tk.Label(self.aydinlatma_islemleri_frame, text="Kontrast Değeri:", bg="#f0f0f0").pack(anchor='w', padx=5)

        self.kontrast_scale_frame = tk.Frame(self.aydinlatma_islemleri_frame, bg="#f0f0f0")
        self.kontrast_scale_frame.pack(fill=tk.X, padx=5, pady=2)

        self.kontrast_deger_label = tk.Label(self.kontrast_scale_frame, text="1.0", width=3, bg="#f0f0f0")
        self.kontrast_deger_label.pack(side=tk.LEFT)

        self.kontrast_scale = Scale(
            self.kontrast_scale_frame,
            from_=0.1,
            to=3.0,
            resolution=0.1,
            orient=HORIZONTAL,
            bg="#f0f0f0",
            command=self.kontrast_degistir
        )
        self.kontrast_scale.set(1.0)
        self.kontrast_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.kontrast_uygula_btn = tk.Button(
            self.aydinlatma_islemleri_frame,
            text="Kontrast Uygula",
            bg="#2196F3",
            fg="white",
            height=1,
            command=self.kontrast_uygula
        )
        self.kontrast_uygula_btn.pack(fill=tk.X, padx=5, pady=2)

        self.kontrast_germe_btn = tk.Button(
            self.aydinlatma_islemleri_frame,
            text="Kontrast Germe",
            bg="#2196F3",
            fg="white",
            height=1,
            command=self.kontrast_germe_uygula
        )
        self.kontrast_germe_btn.pack(fill=tk.X, padx=5, pady=2)

        self.esikleme_islemleri_frame = tk.LabelFrame(self.sol_icerik_frame, text="Eşikleme İşlemleri", bg="#f0f0f0")
        self.esikleme_islemleri_frame.pack(fill=tk.X, padx=10, pady=5)

        tk.Label(self.esikleme_islemleri_frame, text="Eşik Değeri:", bg="#f0f0f0").pack(anchor='w', padx=5)

        self.esik_scale_frame = tk.Frame(self.esikleme_islemleri_frame, bg="#f0f0f0")
        self.esik_scale_frame.pack(fill=tk.X, padx=5, pady=2)

        self.esik_deger_label = tk.Label(self.esik_scale_frame, text="127", width=3, bg="#f0f0f0")
        self.esik_deger_label.pack(side=tk.LEFT)

        self.esik_scale = Scale(self.esik_scale_frame, from_=0, to=255, orient=HORIZONTAL, bg="#f0f0f0",
                                command=self.esik_degeri_degistir)
        self.esik_scale.set(127)
        self.esik_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.ikili_esikleme_btn = tk.Button(self.esikleme_islemleri_frame, text="İkili Eşikleme", bg="#2196F3",
                                            fg="white", height=1, command=self.ikili_esikleme_uygula)
        self.ikili_esikleme_btn.pack(fill=tk.X, padx=5, pady=2)

        self.otsu_esikleme_btn = tk.Button(self.esikleme_islemleri_frame, text="Otsu Eşikleme", bg="#2196F3",
                                           fg="white", height=1, command=self.otsu_esikleme_uygula)
        self.otsu_esikleme_btn.pack(fill=tk.X, padx=5, pady=2)

        self.adaptif_esikleme_btn = tk.Button(self.esikleme_islemleri_frame, text="Adaptif Eşikleme", bg="#2196F3",
                                              fg="white", height=1, command=self.adaptif_esikleme_uygula)
        self.adaptif_esikleme_btn.pack(fill=tk.X, padx=5, pady=2)

        self.filtre_islemleri_frame = tk.LabelFrame(self.sol_icerik_frame, text="Filtre İşlemleri", bg="#f0f0f0")
        self.filtre_islemleri_frame.pack(fill=tk.X, padx=10, pady=5)

        self.yumusatma_frame = tk.Frame(self.filtre_islemleri_frame, bg="#f0f0f0")
        self.yumusatma_frame.pack(fill=tk.X, padx=5, pady=2)

        self.yumusatma_expand = False
        self.yumusatma_header_frame = tk.Frame(self.yumusatma_frame, bg="#f0f0f0")
        self.yumusatma_header_frame.pack(fill=tk.X)

        self.yumusatma_btn = tk.Button(
            self.yumusatma_header_frame,
            text="Yumuşatma Filtreleri ►",
            bg="#2196F3",
            fg="white",
            height=1,
            command=self.toggle_yumusatma_menu
        )
        self.yumusatma_btn.pack(fill=tk.X)
        self.yumusatma_menu_frame = tk.Frame(self.yumusatma_frame, bg="#f0f0f0")
        tk.Label(self.yumusatma_menu_frame, text="Filtre Boyutu:", bg="#f0f0f0").pack(anchor='w', padx=5)

        self.filtre_boyut_frame = tk.Frame(self.yumusatma_menu_frame, bg="#f0f0f0")
        self.filtre_boyut_frame.pack(fill=tk.X, padx=5, pady=2)

        self.filtre_boyut_label = tk.Label(self.filtre_boyut_frame, text="5", width=3, bg="#f0f0f0")
        self.filtre_boyut_label.pack(side=tk.LEFT)

        self.filtre_boyut_scale = Scale(
            self.filtre_boyut_frame,
            from_=3,
            to=25,
            resolution=2,
            orient=HORIZONTAL,
            bg="#f0f0f0",
            command=self.filtre_boyutu_degistir
        )
        self.filtre_boyut_scale.set(5)
        self.filtre_boyut_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.ortalama_filtre_btn = tk.Button(
            self.yumusatma_menu_frame,
            text="Ortalama Filtre Uygula",
            bg="#89CFF0",
            fg="black",
            height=1,
            command=self.ortalama_filtre_uygula
        )
        self.ortalama_filtre_btn.pack(fill=tk.X, padx=5, pady=2)

        self.gauss_filtre_btn = tk.Button(
            self.yumusatma_menu_frame,
            text="Gauss Filtresi Uygula",
            bg="#89CFF0",
            fg="black",
            height=1,
            command=self.gauss_filtre_uygula
        )
        self.gauss_filtre_btn.pack(fill=tk.X, padx=5, pady=2)

        self.medyan_filtre_btn = tk.Button(
            self.yumusatma_menu_frame,
            text="Medyan Filtresi Uygula",
            bg="#89CFF0",
            fg="black",
            height=1,
            command=self.medyan_filtre_uygula
        )
        self.medyan_filtre_btn.pack(fill=tk.X, padx=5, pady=2)

        self.frekans_frame = tk.Frame(self.filtre_islemleri_frame, bg="#f0f0f0")
        self.frekans_frame.pack(fill=tk.X, padx=5, pady=2)

        self.frekans_expand = False
        self.frekans_header_frame = tk.Frame(self.frekans_frame, bg="#f0f0f0")
        self.frekans_header_frame.pack(fill=tk.X)

        self.frekans_btn = tk.Button(
            self.frekans_header_frame,
            text="Frekans Filtreleri ►",
            bg="#2196F3",
            fg="white",
            height=1,
            command=self.toggle_frekans_menu
        )
        self.frekans_btn.pack(fill=tk.X, padx=5, pady=2)
        self.frekans_menu_frame = tk.Frame(self.frekans_frame, bg="#f0f0f0")
        tk.Label(self.frekans_menu_frame, text="Kesme Frekansı:", bg="#f0f0f0").pack(anchor='w', padx=5)

        self.kesme_frekansi_frame = tk.Frame(self.frekans_menu_frame, bg="#f0f0f0")
        self.kesme_frekansi_frame.pack(fill=tk.X, padx=5, pady=2)

        self.kesme_frekansi_label = tk.Label(self.kesme_frekansi_frame, text="30", width=3, bg="#f0f0f0")
        self.kesme_frekansi_label.pack(side=tk.LEFT)

        self.kesme_frekansi_scale = Scale(
            self.kesme_frekansi_frame,
            from_=5,
            to=100,
            resolution=5,
            orient=HORIZONTAL,
            bg="#f0f0f0",
            command=self.kesme_frekansi_degistir
        )
        self.kesme_frekansi_scale.set(30)
        self.kesme_frekansi_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.alcak_geciren_filtre_btn = tk.Button(
            self.frekans_menu_frame,
            text="Alçak Geçiren Filtre",
            bg="#89CFF0",
            fg="black",
            height=1,
            command=self.alcak_geciren_filtre_uygula
        )
        self.alcak_geciren_filtre_btn.pack(fill=tk.X, padx=5, pady=2)

        self.yuksek_geciren_filtre_btn = tk.Button(
            self.frekans_menu_frame,
            text="Yüksek Geçiren Filtre",
            bg="#89CFF0",
            fg="black",
            height=1,
            command=self.yuksek_geciren_filtre_uygula
        )
        self.yuksek_geciren_filtre_btn.pack(fill=tk.X, padx=5, pady=2)

        self.gauss_alcak_geciren_btn = tk.Button(
            self.frekans_menu_frame,
            text="Gauss Alçak Geçiren Filtre",
            bg="#89CFF0",
            fg="black",
            height=1,
            command=self.gauss_alcak_geciren_uygula
        )
        self.gauss_alcak_geciren_btn.pack(fill=tk.X, padx=5, pady=2)

        self.gauss_yuksek_geciren_btn = tk.Button(
            self.frekans_menu_frame,
            text="Gauss Yüksek Geçiren Filtre",
            bg="#89CFF0",
            fg="black",
            height=1,
            command=self.gauss_yuksek_geciren_uygula
        )
        self.gauss_yuksek_geciren_btn.pack(fill=tk.X, padx=5, pady=2)

        self.perspektif_islemleri_frame = tk.LabelFrame(self.sol_icerik_frame, text="Perspektif Düzeltme", bg="#f0f0f0")
        self.perspektif_islemleri_frame.pack(fill=tk.X, padx=10, pady=5)

        self.perspektif_duzeltme_btn = tk.Button(
            self.perspektif_islemleri_frame,
            text="Perspektif Düzeltme",
            bg="#2196F3",
            fg="white",
            height=1,
            command=self.perspektif_duzeltme_uygula
        )
        self.perspektif_duzeltme_btn.pack(fill=tk.X, padx=5, pady=2)

        self.kenar_bulma_frame = tk.LabelFrame(self.sol_icerik_frame, text="Kenar Bulma İşlemleri", bg="#f0f0f0")
        self.kenar_bulma_frame.pack(fill=tk.X, padx=10, pady=5)

        self.kenar_expand = False
        self.kenar_header_frame = tk.Frame(self.kenar_bulma_frame, bg="#f0f0f0")
        self.kenar_header_frame.pack(fill=tk.X)

        self.kenar_btn = tk.Button(
            self.kenar_header_frame,
            text="Kenar Bulma Algoritmaları ►",
            bg="#3F51B5",
            fg="white",
            command=self.toggle_kenar_menu
        )
        self.kenar_btn.pack(fill=tk.X, padx=5, pady=2)

        self.kenar_menu_frame = tk.Frame(self.kenar_bulma_frame, bg="#f0f0f0")

        self.sobel_btn = tk.Button(
            self.kenar_menu_frame,
            text="Sobel Kenar Bulma",
            bg="#7986CB",
            fg="white",
            command=lambda: self.kenar_bulma_uygula("sobel")
        )
        self.sobel_btn.pack(fill=tk.X, padx=5, pady=2)

        self.prewitt_btn = tk.Button(
            self.kenar_menu_frame,
            text="Prewitt Kenar Bulma",
            bg="#7986CB",
            fg="white",
            command=lambda: self.kenar_bulma_uygula("prewitt")
        )
        self.prewitt_btn.pack(fill=tk.X, padx=5, pady=2)

        self.roberts_btn = tk.Button(
            self.kenar_menu_frame,
            text="Roberts Cross Kenar Bulma",
            bg="#7986CB",
            fg="white",
            command=lambda: self.kenar_bulma_uygula("roberts")
        )
        self.roberts_btn.pack(fill=tk.X, padx=5, pady=2)

        self.compass_btn = tk.Button(
            self.kenar_menu_frame,
            text="Compass Kenar Bulma",
            bg="#7986CB",
            fg="white",
            command=lambda: self.kenar_bulma_uygula("compass")
        )
        self.compass_btn.pack(fill=tk.X, padx=5, pady=2)

        self.canny_btn = tk.Button(
            self.kenar_menu_frame,
            text="Canny Kenar Bulma",
            bg="#7986CB",
            fg="white",
            command=lambda: self.kenar_bulma_uygula("canny")
        )
        self.canny_btn.pack(fill=tk.X, padx=5, pady=2)

        self.laplace_btn = tk.Button(
            self.kenar_menu_frame,
            text="Laplace Kenar Bulma",
            bg="#7986CB",
            fg="white",
            command=lambda: self.kenar_bulma_uygula("laplace")
        )
        self.laplace_btn.pack(fill=tk.X, padx=5, pady=2)
        self.gelismis_islemler_frame = tk.LabelFrame(self.sol_icerik_frame, text="Gelişmiş İşlemler", bg="#f0f0f0")
        self.gelismis_islemler_frame.pack(fill=tk.X, padx=10, pady=5)

        self.hough_dogru_btn = tk.Button(
            self.gelismis_islemler_frame,
            text="Hough Doğru Tespiti",
            bg="#673AB7",
            fg="white",
            command=self.hough_dogru_uygula
        )
        self.hough_dogru_btn.pack(fill=tk.X, padx=5, pady=2)

        self.morfolojik_islemler_frame = tk.LabelFrame(self.sol_icerik_frame, text="Morfolojik İşlemler", bg="#f0f0f0")
        self.morfolojik_islemler_frame.pack(fill=tk.X, padx=10, pady=5)

        self.morfolojik_expand = False
        self.morfolojik_header_frame = tk.Frame(self.morfolojik_islemler_frame, bg="#f0f0f0")
        self.morfolojik_header_frame.pack(fill=tk.X)

        self.morfolojik_btn = tk.Button(
            self.morfolojik_header_frame,
            text="Morfolojik İşlemler ►",
            bg="#5D4037",
            fg="white",
            command=self.toggle_morfolojik_menu,
            font=("Arial", 10, "bold"),
            relief=tk.RAISED,
            padx=5,
            pady=3
        )
        self.morfolojik_btn.pack(fill=tk.X, padx=5, pady=2)

        self.morfolojik_canvas = tk.Canvas(self.morfolojik_islemler_frame, bg="#f0f0f0", highlightthickness=0)
        self.morfolojik_scrollbar = tk.Scrollbar(self.morfolojik_islemler_frame, orient="vertical",
                                                 command=self.morfolojik_canvas.yview)
        self.morfolojik_canvas.configure(yscrollcommand=self.morfolojik_scrollbar.set)

        self.morfolojik_scrollable_frame = tk.Frame(self.morfolojik_canvas, bg="#f0f0f0")
        self.morfolojik_scrollable_frame.bind(
            "<Configure>",
            lambda e: self.morfolojik_canvas.configure(scrollregion=self.morfolojik_canvas.bbox("all"))
        )

        self.morfolojik_canvas.create_window((0, 0), window=self.morfolojik_scrollable_frame, anchor="nw", width=260)
        self.morfolojik_menu_frame = tk.Frame(self.morfolojik_scrollable_frame, bg="#f0f0f0")
        self.morfolojik_menu_frame.pack(fill=tk.X, expand=True)

        tk.Label(self.morfolojik_menu_frame, text="Yapısal Eleman Boyutu:", bg="#f0f0f0").pack(anchor='w', padx=5)

        self.kernel_boyut_frame = tk.Frame(self.morfolojik_menu_frame, bg="#f0f0f0")
        self.kernel_boyut_frame.pack(fill=tk.X, padx=5, pady=2)

        self.kernel_boyut_label = tk.Label(self.kernel_boyut_frame, text="3", width=3, bg="#f0f0f0")
        self.kernel_boyut_label.pack(side=tk.LEFT)

        self.kernel_boyut_scale = Scale(
            self.kernel_boyut_frame,
            from_=3,
            to=21,
            resolution=2,
            orient=HORIZONTAL,
            bg="#f0f0f0",
            command=self.kernel_boyutu_degistir
        )
        self.kernel_boyut_scale.set(3)
        self.kernel_boyut_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)

        tk.Label(self.morfolojik_menu_frame, text="Yapısal Eleman Şekli:", bg="#f0f0f0").pack(anchor='w', padx=5)

        self.kernel_sekil_frame = tk.Frame(self.morfolojik_menu_frame, bg="#f0f0f0")
        self.kernel_sekil_frame.pack(fill=tk.X, padx=5, pady=2)

        self.kernel_sekil_var = tk.StringVar()
        self.kernel_sekil_var.set("kare")

        kernel_sekilleri = [("Kare", "kare"), ("Disk", "disk"), ("Çapraz", "çapraz"), ("Elips", "elips")]

        for text, value in kernel_sekilleri:
            rb = tk.Radiobutton(
                self.kernel_sekil_frame,
                text=text,
                variable=self.kernel_sekil_var,
                value=value,
                bg="#f0f0f0"
            )
            rb.pack(side=tk.LEFT, padx=5)

        tk.Label(self.morfolojik_menu_frame, text="İterasyon Sayısı:", bg="#f0f0f0").pack(anchor='w', padx=5)

        self.iterasyon_frame = tk.Frame(self.morfolojik_menu_frame, bg="#f0f0f0")
        self.iterasyon_frame.pack(fill=tk.X, padx=5, pady=2)

        self.iterasyon_label = tk.Label(self.iterasyon_frame, text="1", width=3, bg="#f0f0f0")
        self.iterasyon_label.pack(side=tk.LEFT)

        self.iterasyon_scale = Scale(
            self.iterasyon_frame,
            from_=1,
            to=10,
            orient=HORIZONTAL,
            bg="#f0f0f0",
            command=self.iterasyon_degistir
        )
        self.iterasyon_scale.set(1)
        self.iterasyon_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.erode_btn = tk.Button(
            self.morfolojik_menu_frame,
            text="Aşındırma (Erode)",
            bg="#8D6E63",
            fg="white",
            command=self.erode_uygula,
            font=("Arial", 9, "bold"),
            relief=tk.RAISED,
            padx=3,
            pady=2
        )
        self.erode_btn.pack(fill=tk.X, padx=5, pady=3)

        self.dilate_btn = tk.Button(
            self.morfolojik_menu_frame,
            text="Genişletme (Dilate)",
            bg="#8D6E63",
            fg="white",
            command=self.dilate_uygula,
            font=("Arial", 9, "bold"),
            relief=tk.RAISED,
            padx=3,
            pady=2
        )
        self.dilate_btn.pack(fill=tk.X, padx=5, pady=3)

        self.opening_btn = tk.Button(
            self.morfolojik_menu_frame,
            text="Açma (Opening)",
            bg="#8D6E63",
            fg="white",
            command=self.opening_uygula,
            font=("Arial", 9, "bold"),
            relief=tk.RAISED,
            padx=3,
            pady=2
        )
        self.opening_btn.pack(fill=tk.X, padx=5, pady=3)

        self.closing_btn = tk.Button(
            self.morfolojik_menu_frame,
            text="Kapama (Closing)",
            bg="#8D6E63",
            fg="white",
            command=self.closing_uygula,
            font=("Arial", 9, "bold"),
            relief=tk.RAISED,
            padx=3,
            pady=2
        )
        self.closing_btn.pack(fill=tk.X, padx=5, pady=3)

        self.kernel_goster_btn = tk.Button(
            self.morfolojik_menu_frame,
            text="Yapısal Elemanları Göster",
            bg="#8D6E63",
            fg="white",
            command=self.kernel_goster_uygula,
            font=("Arial", 9, "bold"),
            relief=tk.RAISED,
            padx=3,
            pady=2
        )
        self.kernel_goster_btn.pack(fill=tk.X, padx=5, pady=3)


        self.hough_cember_btn = tk.Button(
            self.gelismis_islemler_frame,
            text="Hough Çember Tespiti",
            bg="#673AB7",
            fg="white",
            command=self.hough_cember_uygula
        )
        self.hough_cember_btn.pack(fill=tk.X, padx=5, pady=2)

        self.gabor_filtre_btn = tk.Button(
            self.gelismis_islemler_frame,
            text="Gabor Filtresi",
            bg="#673AB7",
            fg="white",
            command=self.gabor_filtre_uygula
        )
        self.gabor_filtre_btn.pack(fill=tk.X, padx=5, pady=2)

        self.kmeans_btn = tk.Button(
            self.gelismis_islemler_frame,
            text="K-Means Segmentasyon",
            bg="#673AB7",
            fg="white",
            command=self.kmeans_uygula
        )
        self.kmeans_btn.pack(fill=tk.X, padx=5, pady=2)

        self.histogram_islemleri_frame = tk.LabelFrame(self.sol_icerik_frame, text="Histogram İşlemleri", bg="#f0f0f0")
        self.histogram_islemleri_frame.pack(fill=tk.X, padx=10, pady=5)

        self.histogram_goster_btn = tk.Button(
            self.histogram_islemleri_frame,
            text="Histogram Göster",
            bg="#2196F3",
            fg="white",
            height=1,
            command=self.histogram_goster
        )
        self.histogram_goster_btn.pack(fill=tk.X, padx=5, pady=2)

        self.histogram_esitle_btn = tk.Button(
            self.histogram_islemleri_frame,
            text="Histogram Eşitleme",
            bg="#2196F3",
            fg="white",
            height=1,
            command=self.histogram_esitle
        )
        self.histogram_esitle_btn.pack(fill=tk.X, padx=5, pady=2)

        self.sag_panel = tk.Frame(self.main_frame, bg="black")
        self.sag_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)


        self.goruntu_mesaji = tk.Label(self.sag_panel, text="Lütfen bir görüntü yükleyin",
                                       bg="black", fg="white", font=("Arial", 16))
        self.goruntu_mesaji.pack(fill=tk.BOTH, expand=True)

        self.goruntu_label = tk.Label(self.sag_panel, bg="black")
        self.goruntu_label.pack_forget()

        self.goruntu_label.bind("<Button-1>", self.on_mouse_down)
        self.goruntu_label.bind("<B1-Motion>", self.on_mouse_move)
        self.goruntu_label.bind("<ButtonRelease-1>", self.on_mouse_up)


        self.original_image = None
        self.goruntu_label = tk.Label(self.sag_panel, bg="black")

        self.original_image = None
        self.current_image = None
        self.photo_image = None
        self.working_image = None

        self.sol_icerik_frame.update_idletasks()
        total_height = self.sol_icerik_frame.winfo_reqheight() + 300  # İçeriğin altında ekstra boşluk bırak
        self.sol_canvas.config(scrollregion=(0, 0, self.sol_icerik_frame.winfo_reqwidth(), total_height))


        self.sol_canvas.bind_all("<MouseWheel>", self._on_mousewheel)


        self.sol_canvas.bind("<Configure>", self._on_canvas_configure)

    def _on_mousewheel(self, event):
        if self.morfolojik_expand:
            x, y = event.x_root, event.y_root
            canvas_x = self.morfolojik_canvas.winfo_rootx()
            canvas_y = self.morfolojik_canvas.winfo_rooty()
            canvas_width = self.morfolojik_canvas.winfo_width()
            canvas_height = self.morfolojik_canvas.winfo_height()

            if (canvas_x <= x <= canvas_x + canvas_width and
                    canvas_y <= y <= canvas_y + canvas_height):
                return

        self.sol_canvas.yview_scroll(int(-3 * (event.delta / 120)), "units")

    def _on_canvas_configure(self, event):
        self.sol_canvas.itemconfig(self.sol_canvas_window, width=event.width)

        self.sol_icerik_frame.update_idletasks()
        total_height = self.sol_icerik_frame.winfo_reqheight() + 500  # Daha fazla scroll alanı
        self.sol_canvas.config(scrollregion=(0, 0, event.width, total_height))

    def goruntu_yukle(self):
        try:
            initial_dir = os.path.join(os.path.expanduser("~"), "Desktop")
            if not os.path.exists(initial_dir):
                initial_dir = os.path.join(os.path.expanduser("~"), "Documents")
            if not os.path.exists(initial_dir):
                initial_dir = os.path.expanduser("~")

            filepath = filedialog.askopenfilename(
                title="Görüntü Seç",
                initialdir=initial_dir,
                filetypes=(
                    ("Görüntü Dosyaları", "*.png *.jpg *.jpeg *.bmp *.gif *.tif *.tiff"),
                    ("JPEG Dosyaları", "*.jpg *.jpeg"),
                    ("PNG Dosyaları", "*.png"),
                    ("BMP Dosyaları", "*.bmp"),
                    ("TIFF Dosyaları", "*.tif *.tiff"),
                    ("Tüm Dosyalar", "*.*")
                )
            )

            if filepath:
                self.original_image = cv2.imread(filepath)

                if self.original_image is not None:
                    self.current_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
                    self.working_image = self.current_image.copy()
                    self.goruntu_guncelle()
                    self.goruntu_mesaji.pack_forget()
                    self.goruntu_label.pack(fill=tk.BOTH, expand=True)
                else:
                    self.goruntu_mesaji.config(text="Görüntü yüklenemedi!")
        except Exception as e:
            print(f"Hata: Dosya dialog penceresi açılırken bir hata oluştu: {str(e)}")

    def orijinal_goruntu_don(self):
        if self.original_image is None:
            print("Hata: Orijinal görüntü yok!")
            return

        try:
            self.current_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
            self.working_image = self.current_image.copy()
            self.goruntu_guncelle()

        except Exception as e:
            print(f"Hata: Orijinal görüntüye dönülürken bir hata oluştu: {str(e)}")

    def goruntu_kaydet(self):
        if self.working_image is not None:
            try:
                initial_dir = os.path.join(os.path.expanduser("~"), "Desktop")
                if not os.path.exists(initial_dir):
                    initial_dir = os.path.join(os.path.expanduser("~"), "Documents")

                filepath = filedialog.asksaveasfilename(
                    title="Görüntüyü Kaydet",
                    initialdir=initial_dir,
                    defaultextension=".jpg",
                    filetypes=(
                        ("JPEG Dosyaları", "*.jpg"),
                        ("PNG Dosyaları", "*.png"),
                        ("BMP Dosyaları", "*.bmp"),
                        ("TIFF Dosyaları", "*.tif")
                    )
                )

                if filepath:
                    _, ext = os.path.splitext(filepath)
                    ext = ext.lower()

                    save_image = cv2.cvtColor(self.working_image, cv2.COLOR_RGB2BGR)

                    if ext == ".jpg" or ext == ".jpeg":
                        cv2.imwrite(filepath, save_image, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    elif ext == ".png":
                        cv2.imwrite(filepath, save_image, [cv2.IMWRITE_PNG_COMPRESSION, 9])
                    else:
                        cv2.imwrite(filepath, save_image)

                    print(f"Bilgi: Görüntü başarıyla kaydedildi: {filepath}")
            except Exception as e:
                print(f"Hata: Görüntü kaydedilirken bir hata oluştu: {str(e)}")

    def goruntu_guncelle(self):
        if self.working_image is not None:
            try:
                panel_width = self.sag_panel.winfo_width()
                panel_height = self.sag_panel.winfo_height()

                if panel_width <= 1:
                    panel_width = self.root.winfo_width() - self.sol_panel.winfo_width()
                if panel_height <= 1:
                    panel_height = self.root.winfo_height()

                img_height, img_width = self.working_image.shape[:2]

                scale = min(panel_width / img_width, panel_height / img_height)

                if scale < 1 or scale > 1:
                    new_width = int(img_width * scale)
                    new_height = int(img_height * scale)
                    resized_image = cv2.resize(self.working_image, (new_width, new_height),
                                               interpolation=cv2.INTER_AREA)
                else:
                    resized_image = self.working_image.copy()

                pil_image = Image.fromarray(resized_image)
                self.photo_image = ImageTk.PhotoImage(image=pil_image)

                self.goruntu_label.config(image=self.photo_image)
                self.goruntu_label.image = self.photo_image  # Referansı koru
            except Exception as e:
                print(f"Hata: Görüntü güncellenirken bir hata oluştu: {str(e)}")

    def griye_cevir_uygula(self):
        if self.working_image is None:
            print("Hata: İşlem yapılacak bir görüntü yok!")
            return

        try:
            bgr_image = cv2.cvtColor(self.working_image, cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
            self.working_image = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
            self.goruntu_guncelle()

        except Exception as e:
            print(f"Hata: Gri tonlama işlemi uygulanırken bir hata oluştu: {str(e)}")

    def negatife_cevir_uygula(self):
        if self.working_image is None:
            print("Hata: İşlem yapılacak bir görüntü yok!")
            return

        try:
            self.working_image = 255 - self.working_image
            self.goruntu_guncelle()

        except Exception as e:
            print(f"Hata: Negatif işlemi uygulanırken bir hata oluştu: {str(e)}")

    def kanal_goster_uygula(self, kanal):
        if self.working_image is None:
            print("Hata: İşlem yapılacak bir görüntü yok!")
            return

        try:
            bgr_image = cv2.cvtColor(self.working_image, cv2.COLOR_RGB2BGR)

            b, g, r = cv2.split(bgr_image)
            zeros = np.zeros_like(b)

            if kanal.lower() == 'blue':
                result = cv2.merge([b, zeros, zeros])
            elif kanal.lower() == 'green':
                result = cv2.merge([zeros, g, zeros])
            elif kanal.lower() == 'red':
                result = cv2.merge([zeros, zeros, r])
            else:
                print("Hata: Geçersiz kanal seçimi! (blue, green veya red olmalı)")
                return

            self.working_image = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
            self.goruntu_guncelle()

        except Exception as e:
            print(f"Hata: Renk kanalı işlemi uygulanırken bir hata oluştu: {str(e)}")

    def parlaklik_degistir(self, val):
        try:
            deger = int(float(val))
            self.parlaklik_deger_label.config(text=str(deger))
        except Exception as e:
            print(f"Hata: Parlaklık değiştirilirken bir hata oluştu: {str(e)}")

    def parlaklik_uygula(self):
        try:
            deger = int(self.parlaklik_scale.get())

            if self.working_image is not None:
                self.working_image = self.parlaklik_ayarla_hizli(self.working_image, deger)
                self.goruntu_guncelle()
                self.parlaklik_scale.set(0)
                self.parlaklik_deger_label.config(text="0")
        except Exception as e:
            print(f"Hata: Parlaklık uygulanırken bir hata oluştu: {str(e)}")

    def kontrast_degistir(self, val):
        try:
            deger = float(val)
            self.kontrast_deger_label.config(text=f"{deger:.1f}")
        except Exception as e:
            print(f"Hata: Kontrast değiştirilirken bir hata oluştu: {str(e)}")

    def kontrast_uygula(self):
        try:
            deger = float(self.kontrast_scale.get())

            if self.working_image is not None:
                self.working_image = self.kontrast_ayarla(self.working_image, deger)
                self.goruntu_guncelle()
                self.kontrast_scale.set(1.0)
                self.kontrast_deger_label.config(text="1.0")
        except Exception as e:
            print(f"Hata: Kontrast uygulanırken bir hata oluştu: {str(e)}")

    def parlaklik_ayarla_hizli(self, img, deger):

        if img is None:
            print("Hata: İşlem yapılacak bir görüntü yok!")
            return None

        try:
            sonuc = np.clip(img.astype(np.int16) + deger, 0, 255).astype(np.uint8)
            return sonuc

        except Exception as e:
            print(f"Hata: Parlaklık ayarlanırken bir hata oluştu: {str(e)}")
            return img

    def kontrast_ayarla(self, img, alpha, beta=0):

        if img is None:
            print("Hata: İşlem yapılacak bir görüntü yok!")
            return None

        try:
            # Formül: g(x,y) = alpha * f(x,y) + beta
            sonuc = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
            return sonuc

        except Exception as e:
            print(f"Hata: Kontrast ayarlama işlemi uygulanırken bir hata oluştu: {str(e)}")
            return img

    def kontrast_germe_uygula(self):
        if self.working_image is None:
            print("Hata: İşlem yapılacak bir görüntü yok!")
            return

        try:
            b, g, r = cv2.split(self.working_image)
            b_stretched = self.kontrast_germe_kanali(b)
            g_stretched = self.kontrast_germe_kanali(g)
            r_stretched = self.kontrast_germe_kanali(r)

            self.working_image = cv2.merge([b_stretched, g_stretched, r_stretched])
            self.goruntu_guncelle()

        except Exception as e:
            print(f"Hata: Kontrast germe işlemi uygulanırken bir hata oluştu: {str(e)}")

    def kontrast_germe_kanali(self, kanal):
        min_val = np.min(kanal)
        max_val = np.max(kanal)

        # (x - min) / (max - min) * 255
        if max_val > min_val:
            return ((kanal.astype(np.float32) - min_val) / (max_val - min_val) * 255).astype(np.uint8)
        else:
            return kanal

    def esik_degeri_degistir(self, val):
        try:
            deger = int(float(val))
            self.esik_deger_label.config(text=str(deger))
        except Exception as e:
            print(f"Hata: Eşik değeri değiştirilirken bir hata oluştu: {str(e)}")

    def ikili_esikleme_uygula(self):
        if self.working_image is None:
            print("Hata: İşlem yapılacak bir görüntü yok!")
            return

        try:
            bgr_img = cv2.cvtColor(self.working_image, cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
            esik_degeri = self.esik_scale.get()
            _, sonuc = cv2.threshold(gray, esik_degeri, 255, cv2.THRESH_BINARY)
            self.working_image = cv2.cvtColor(sonuc, cv2.COLOR_GRAY2RGB)
            self.goruntu_guncelle()

        except Exception as e:
            print(f"Hata: İkili eşikleme işlemi uygulanırken bir hata oluştu: {str(e)}")

    def otsu_esikleme_uygula(self):
        if self.working_image is None:
            print("Hata: İşlem yapılacak bir görüntü yok!")
            return

        try:
            bgr_img = cv2.cvtColor(self.working_image, cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)

            _, sonuc = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            otsu_esik = int(_)
            self.esik_scale.set(otsu_esik)
            self.esik_deger_label.config(text=str(otsu_esik))

            self.working_image = cv2.cvtColor(sonuc, cv2.COLOR_GRAY2RGB)
            self.goruntu_guncelle()

        except Exception as e:
            print(f"Hata: Otsu eşikleme işlemi uygulanırken bir hata oluştu: {str(e)}")

    def adaptif_esikleme_uygula(self):
        if self.working_image is None:
            print("Hata: İşlem yapılacak bir görüntü yok!")
            return

        try:
            bgr_img = cv2.cvtColor(self.working_image, cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)

            sonuc = cv2.adaptiveThreshold(
                gray,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                11,
                2
            )

            self.working_image = cv2.cvtColor(sonuc, cv2.COLOR_GRAY2RGB)
            self.goruntu_guncelle()

        except Exception as e:
            print(f"Hata: Adaptif eşikleme işlemi uygulanırken bir hata oluştu: {str(e)}")

    def histogram_goster(self):
        if self.working_image is None:
            print("Hata: İşlem yapılacak bir görüntü yok!")
            return

        try:
            histogram_pencere = tk.Toplevel(self.root)
            histogram_pencere.title("Görüntü Histogramı")
            histogram_pencere.geometry("600x400")

            fig = plt.Figure(figsize=(6, 4), dpi=100)
            canvas = FigureCanvasTkAgg(fig, master=histogram_pencere)
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

            b, g, r = cv2.split(self.working_image)

            ax = fig.add_subplot(111)

            bins = np.arange(0, 257, 1)
            ax.hist(r.ravel(), bins=bins, alpha=0.5, color='red', label='R')
            ax.hist(g.ravel(), bins=bins, alpha=0.5, color='green', label='G')
            ax.hist(b.ravel(), bins=bins, alpha=0.5, color='blue', label='B')

            ax.set_title("RGB Histogram")
            ax.set_xlabel("Piksel Değeri")
            ax.set_ylabel("Frekans")
            ax.set_xlim([0, 256])
            ax.legend()

            fig.tight_layout()
            canvas.draw()

        except Exception as e:
            print(f"Hata: Histogram gösterilirken bir hata oluştu: {str(e)}")

    def histogram_esitle(self):
        if self.working_image is None:
            print("Hata: İşlem yapılacak bir görüntü yok!")
            return

        try:
            ycrcb = cv2.cvtColor(self.working_image, cv2.COLOR_RGB2YCrCb)

            y, cr, cb = cv2.split(ycrcb)

            y_eq = cv2.equalizeHist(y)

            ycrcb_eq = cv2.merge([y_eq, cr, cb])

            self.working_image = cv2.cvtColor(ycrcb_eq, cv2.COLOR_YCrCb2RGB)
            self.goruntu_guncelle()

        except Exception as e:
            print(f"Hata: Histogram eşitleme işlemi uygulanırken bir hata oluştu: {str(e)}")

    def dondurme_degeri_degistir(self, val):
        try:
            deger = int(float(val))
            self.dondurme_deger_label.config(text=f"{deger}°")
        except Exception as e:
            print(f"Hata: Döndürme değeri değiştirilirken bir hata oluştu: {str(e)}")

    def dondurme_uygula(self):
        if self.working_image is None:
            print("Hata: İşlem yapılacak bir görüntü yok!")
            return

        try:
            aci = int(self.dondurme_scale.get())
            bgr_img = cv2.cvtColor(self.working_image, cv2.COLOR_RGB2BGR)
            h, w = bgr_img.shape[:2]
            merkez = (w // 2, h // 2)
            rotation_matrix = cv2.getRotationMatrix2D(merkez, aci, 1.0)
            donmus = cv2.warpAffine(bgr_img, rotation_matrix, (w, h))
            self.working_image = cv2.cvtColor(donmus, cv2.COLOR_BGR2RGB)
            self.goruntu_guncelle()
            self.dondurme_scale.set(0)
            self.dondurme_deger_label.config(text="0°")

        except Exception as e:
            print(f"Hata: Döndürme işlemi uygulanırken bir hata oluştu: {str(e)}")

    def olcekleme_degeri_degistir(self, val):
        try:
            deger = float(val)
            self.olcekleme_deger_label.config(text=f"{deger:.1f}")
        except Exception as e:
            print(f"Hata: Ölçekleme değeri değiştirilirken bir hata oluştu: {str(e)}")

    def olcekleme_uygula(self):
        if self.working_image is None:
            print("Hata: İşlem yapılacak bir görüntü yok!")
            return

        try:
            skala = float(self.olcekleme_scale.get())
            bgr_img = cv2.cvtColor(self.working_image, cv2.COLOR_RGB2BGR)
            h, w = bgr_img.shape[:2]
            new_w = int(w * skala)
            new_h = int(h * skala)
            olceklenmis = cv2.resize(bgr_img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            self.working_image = cv2.cvtColor(olceklenmis, cv2.COLOR_BGR2RGB)
            self.goruntu_guncelle()
            self.olcekleme_scale.set(1.0)
            self.olcekleme_deger_label.config(text="1.0")

        except Exception as e:
            print(f"Hata: Ölçekleme işlemi uygulanırken bir hata oluştu: {str(e)}")

    def tasima_uygula(self):
        if self.working_image is None:
            print("Hata: İşlem yapılacak bir görüntü yok!")
            return

        try:
            try:
                dx = int(self.tasima_x_entry.get())
                dy = int(self.tasima_y_entry.get())
            except ValueError:
                print("Hata: Geçersiz taşıma değerleri! Sayısal değer giriniz.")
                return

            bgr_img = cv2.cvtColor(self.working_image, cv2.COLOR_RGB2BGR)
            h, w = bgr_img.shape[:2]
            T = np.float32([[1, 0, dx], [0, 1, dy]])
            tasinmis = cv2.warpAffine(bgr_img, T, (w, h))
            self.working_image = cv2.cvtColor(tasinmis, cv2.COLOR_BGR2RGB)
            self.goruntu_guncelle()
            self.tasima_x_entry.delete(0, tk.END)
            self.tasima_x_entry.insert(0, "0")
            self.tasima_y_entry.delete(0, tk.END)
            self.tasima_y_entry.insert(0, "0")

        except Exception as e:
            print(f"Hata: Taşıma işlemi uygulanırken bir hata oluştu: {str(e)}")

    def egme_x_uygula(self):
        if self.working_image is None:
            print("Hata: İşlem yapılacak bir görüntü yok!")
            return

        try:
            sh_x = 0.5
            bgr_img = cv2.cvtColor(self.working_image, cv2.COLOR_RGB2BGR)
            h, w = bgr_img.shape[:2]
            S = np.float32([[1, sh_x, 0], [0, 1, 0]])
            new_w = w + int(abs(sh_x) * h)
            sheared = cv2.warpAffine(bgr_img, S, (new_w, h))
            self.working_image = cv2.cvtColor(sheared, cv2.COLOR_BGR2RGB)
            self.goruntu_guncelle()

        except Exception as e:
            print(f"Hata: X ekseninde eğme işlemi uygulanırken bir hata oluştu: {str(e)}")

    def egme_y_uygula(self):
        if self.working_image is None:
            print("Hata: İşlem yapılacak bir görüntü yok!")
            return

        try:
            sh_y = 0.5
            bgr_img = cv2.cvtColor(self.working_image, cv2.COLOR_RGB2BGR)
            h, w = bgr_img.shape[:2]
            S = np.float32([[1, 0, 0], [sh_y, 1, 0]])
            new_h = h + int(abs(sh_y) * w)
            sheared = cv2.warpAffine(bgr_img, S, (w, new_h))
            self.working_image = cv2.cvtColor(sheared, cv2.COLOR_BGR2RGB)
            self.goruntu_guncelle()

        except Exception as e:
            print(f"Hata: Y ekseninde eğme işlemi uygulanırken bir hata oluştu: {str(e)}")

    def aynalama_dikey_uygula(self):
        if self.working_image is None:
            print("Hata: İşlem yapılacak bir görüntü yok!")
            return

        try:
            bgr_img = cv2.cvtColor(self.working_image, cv2.COLOR_RGB2BGR)
            mirrored = cv2.flip(bgr_img, 1)
            self.working_image = cv2.cvtColor(mirrored, cv2.COLOR_BGR2RGB)
            self.goruntu_guncelle()

        except Exception as e:
            print(f"Hata: Dikey eksende aynalama işlemi uygulanırken bir hata oluştu: {str(e)}")

    def aynalama_yatay_uygula(self):
        if self.working_image is None:
            print("Hata: İşlem yapılacak bir görüntü yok!")
            return

        try:
            bgr_img = cv2.cvtColor(self.working_image, cv2.COLOR_RGB2BGR)
            mirrored = cv2.flip(bgr_img, 0)
            self.working_image = cv2.cvtColor(mirrored, cv2.COLOR_BGR2RGB)
            self.goruntu_guncelle()

        except Exception as e:
            print(f"Hata: Yatay eksende aynalama işlemi uygulanırken bir hata oluştu: {str(e)}")

    def aynalama_her_iki_uygula(self):
        if self.working_image is None:
            print("Hata: İşlem yapılacak bir görüntü yok!")
            return

        try:
            bgr_img = cv2.cvtColor(self.working_image, cv2.COLOR_RGB2BGR)
            mirrored = cv2.flip(bgr_img, -1)
            self.working_image = cv2.cvtColor(mirrored, cv2.COLOR_BGR2RGB)
            self.goruntu_guncelle()

        except Exception as e:
            print(f"Hata: Her iki eksende aynalama işlemi uygulanırken bir hata oluştu: {str(e)}")

    def kirpma_uygula(self):
        if self.working_image is None:
            print("Hata: İşlem yapılacak bir görüntü yok!")
            return
        try:
            print("Not: Kırpma işlemi için görüntü üzerinde seçim yapmanız gerekecek.")
            print("OpenCV penceresi açılacak ve fare ile seçim yapabileceksiniz.")
            print("Seçim yaptıktan sonra herhangi bir tuşa basarak işlemi tamamlayabilirsiniz.")

            bgr_img = cv2.cvtColor(self.working_image, cv2.COLOR_RGB2BGR)
            cv2.namedWindow("Kırpma için seçim yapın", cv2.WINDOW_NORMAL)
            h, w = bgr_img.shape[:2]
            cv2.resizeWindow("Kırpma için seçim yapın", w, h)
            roi = cv2.selectROI("Kırpma için seçim yapın", bgr_img, False)
            cv2.destroyAllWindows()
            x, y, w, h = roi

            if w > 0 and h > 0:
                cropped = bgr_img[y:y + h, x:x + w]
                self.working_image = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
                self.goruntu_guncelle()
            else:
                print("Geçerli bir kırpma bölgesi seçilmedi!")

        except Exception as e:
            print(f"Hata: Kırpma işlemi uygulanırken bir hata oluştu: {str(e)}")

    def perspektif_duzeltme_uygula(self):
        if self.working_image is None:
            print("Hata: İşlem yapılacak bir görüntü yok!")
            return

        try:
            bgr_img = cv2.cvtColor(self.working_image, cv2.COLOR_RGB2BGR)
            print("Not: Perspektif düzeltme işlemi için görüntü üzerinde 4 nokta seçmeniz gerekecek.")
            print("OpenCV penceresi açılacak ve fare ile nokta seçimi yapabileceksiniz.")
            print("Lütfen şu sırayla noktaları seçin: Sol üst, Sağ üst, Sol alt, Sağ alt")
            print("İşlemi tamamlamak için R tuşuna, iptal etmek için ESC tuşuna basın.")
            print("Noktaları sıfırlamak için C tuşuna basın.")

            secilen_noktalar = []
            calisma_kopya = bgr_img.copy()

            def nokta_sec(event, x, y, flags, param):
                nonlocal secilen_noktalar, calisma_kopya

                if event == cv2.EVENT_LBUTTONDOWN:

                    if len(secilen_noktalar) < 4:
                        secilen_noktalar.append((x, y))
                        print(f"Nokta {len(secilen_noktalar)} Seçildi: {x}, {y}")

                        calisma_kopya = bgr_img.copy()
                        for i, (px, py) in enumerate(secilen_noktalar):
                            cv2.circle(calisma_kopya, (px, py), 5, (0, 0, 255), -1)
                            cv2.putText(calisma_kopya, f"{i + 1}", (px + 10, py + 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                        if len(secilen_noktalar) >= 2:
                            for i in range(len(secilen_noktalar) - 1):
                                pt1 = secilen_noktalar[i]
                                pt2 = secilen_noktalar[i + 1]
                                cv2.line(calisma_kopya, pt1, pt2, (255, 0, 0), 2)


                        if len(secilen_noktalar) == 4:
                            cv2.line(calisma_kopya, secilen_noktalar[3], secilen_noktalar[0], (255, 0, 0), 2)

                        cv2.imshow(pencere_adi, calisma_kopya)

            pencere_adi = "Perspektif Düzeltme - 4 nokta seçin"
            cv2.namedWindow(pencere_adi, cv2.WINDOW_NORMAL)
            h, w = bgr_img.shape[:2]
            cv2.resizeWindow(pencere_adi, w, h)
            cv2.setMouseCallback(pencere_adi, nokta_sec)
            cv2.imshow(pencere_adi, bgr_img)

            while True:
                key = cv2.waitKey(0) & 0xFF

                # ESC ile iptal
                if key == 27:
                    print("İşlem iptal edildi.")
                    cv2.destroyWindow(pencere_adi)
                    return

                # C ile temizle
                elif key == ord('c') or key == ord('C'):
                    secilen_noktalar = []
                    calisma_kopya = bgr_img.copy()
                    cv2.imshow(pencere_adi, calisma_kopya)
                    print("Seçilen noktalar temizlendi. Tekrar seçim yapabilirsiniz.")

                # R ile tamamla
                elif key == ord('r') or key == ord('R'):
                    if len(secilen_noktalar) == 4:
                        break
                    else:
                        print(f"Lütfen 4 nokta seçin! Şu ana kadar {len(secilen_noktalar)} nokta seçildi.")


            cv2.destroyWindow(pencere_adi)
            if len(secilen_noktalar) == 4:
                kaynak_noktalar = np.float32(secilen_noktalar)
                hedef_genislik, hedef_yukseklik = 500, 500
                hedef_noktalar = np.float32([
                    [0, 0],  # Sol üst
                    [hedef_genislik, 0],  # Sağ üst
                    [0, hedef_yukseklik],  # Sol alt
                    [hedef_genislik, hedef_yukseklik]  # Sağ alt
                ])
                matrix = cv2.getPerspectiveTransform(kaynak_noktalar, hedef_noktalar)
                duzeltilmis_goruntu = cv2.warpPerspective(bgr_img, matrix, (hedef_genislik, hedef_yukseklik))
                self.working_image = cv2.cvtColor(duzeltilmis_goruntu, cv2.COLOR_BGR2RGB)
                self.goruntu_guncelle()
                print("Perspektif düzeltme işlemi başarıyla tamamlandı.")

        except Exception as e:
            print(f"Hata: Perspektif düzeltme işlemi uygulanırken bir hata oluştu: {str(e)}")
            cv2.destroyAllWindows()

    def toggle_yumusatma_menu(self):
        if self.yumusatma_expand:
            self.yumusatma_menu_frame.pack_forget()
            self.yumusatma_btn.config(text="Yumuşatma Filtreleri ►")
            self.yumusatma_expand = False
        else:
            self.yumusatma_menu_frame.pack(fill=tk.X, padx=5, pady=2)
            self.yumusatma_btn.config(text="Yumuşatma Filtreleri ▼")
            self.yumusatma_expand = True

    def toggle_frekans_menu(self):
        if self.frekans_expand:
            self.frekans_menu_frame.pack_forget()
            self.frekans_btn.config(text="Frekans Filtreleri ►")
            self.frekans_expand = False
        else:
            self.frekans_menu_frame.pack(fill=tk.X, padx=5, pady=2)
            self.frekans_btn.config(text="Frekans Filtreleri ▼")
            self.frekans_expand = True

    def filtre_boyutu_degistir(self, val):
        try:
            deger = int(float(val))
            if deger % 2 == 0:
                deger += 1
                self.filtre_boyut_scale.set(deger)
            self.filtre_boyut_label.config(text=str(deger))
        except Exception as e:
            print(f"Hata: Filtre boyutu değiştirilirken bir hata oluştu: {str(e)}")

    def kesme_frekansi_degistir(self, val):
        try:
            deger = int(float(val))
            self.kesme_frekansi_label.config(text=str(deger))
        except Exception as e:
            print(f"Hata: Kesme frekansı değiştirilirken bir hata oluştu: {str(e)}")

    def ortalama_filtre_uygula(self):
        if self.working_image is None:
            print("Hata: İşlem yapılacak bir görüntü yok!")
            return

        try:
            kernel_size = self.filtre_boyut_scale.get()
            bgr_img = cv2.cvtColor(self.working_image, cv2.COLOR_RGB2BGR)
            filtered = cv2.blur(bgr_img, (kernel_size, kernel_size))

            self.working_image = cv2.cvtColor(filtered, cv2.COLOR_BGR2RGB)
            self.goruntu_guncelle()
            print(f"Ortalama filtresi uygulandı (Boyut: {kernel_size}x{kernel_size})")

        except Exception as e:
            print(f"Hata: Ortalama filtresi uygulanırken bir hata oluştu: {str(e)}")

    def gauss_filtre_uygula(self):
        if self.working_image is None:
            print("Hata: İşlem yapılacak bir görüntü yok!")
            return

        try:
            kernel_size = self.filtre_boyut_scale.get()
            sigma = kernel_size / 6.0
            bgr_img = cv2.cvtColor(self.working_image, cv2.COLOR_RGB2BGR)
            filtered = cv2.GaussianBlur(bgr_img, (kernel_size, kernel_size), sigma)
            self.working_image = cv2.cvtColor(filtered, cv2.COLOR_BGR2RGB)
            self.goruntu_guncelle()
            print(f"Gauss filtresi uygulandı (Boyut: {kernel_size}x{kernel_size}, Sigma: {sigma:.2f})")

        except Exception as e:
            print(f"Hata: Gauss filtresi uygulanırken bir hata oluştu: {str(e)}")

    def medyan_filtre_uygula(self):
        if self.working_image is None:
            print("Hata: İşlem yapılacak bir görüntü yok!")
            return

        try:
            kernel_size = self.filtre_boyut_scale.get()
            bgr_img = cv2.cvtColor(self.working_image, cv2.COLOR_RGB2BGR)
            filtered = cv2.medianBlur(bgr_img, kernel_size)
            self.working_image = cv2.cvtColor(filtered, cv2.COLOR_BGR2RGB)
            self.goruntu_guncelle()
            print(f"Medyan filtresi uygulandı (Boyut: {kernel_size}x{kernel_size})")

        except Exception as e:
            print(f"Hata: Medyan filtresi uygulanırken bir hata oluştu: {str(e)}")

    def fft_goruntu(self, img):
        if img is None:
            print("Hata: İşlenecek görüntü bulunamadı!")
            return None, None

        try:

            if len(img.shape) > 2:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                gray = img.copy()

            f_transform = np.fft.fft2(gray)
            f_transform_shifted = np.fft.fftshift(f_transform)
            magnitude_spectrum = 20 * np.log(np.abs(f_transform_shifted) + 1)

            return f_transform_shifted, magnitude_spectrum

        except Exception as e:
            print(f"Hata: Fourier dönüşümü hesaplanırken bir hata oluştu: {str(e)}")
            return None, None

    def alcak_geciren_filtre_uygula(self):

        if self.working_image is None:
            print("Hata: İşlem yapılacak bir görüntü yok!")
            return

        try:
            kesme_frekansi = self.kesme_frekansi_scale.get()
            bgr_img = cv2.cvtColor(self.working_image, cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
            f_transform_shifted, _ = self.fft_goruntu(gray)

            if f_transform_shifted is None:
                print("Hata: Fourier dönüşümü hesaplanamadı!")
                return

            rows, cols = gray.shape
            mask = np.zeros((rows, cols), np.uint8)
            center = (cols // 2, rows // 2)
            cv2.circle(mask, center, kesme_frekansi, 1, -1)
            filtered_fft = f_transform_shifted * mask
            filtered_image = np.fft.ifft2(np.fft.ifftshift(filtered_fft)).real
            filtered_image = cv2.normalize(filtered_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            self.working_image = cv2.cvtColor(filtered_image, cv2.COLOR_GRAY2RGB)
            self.goruntu_guncelle()
            print(f"Alçak geçiren filtre uygulandı (Kesme frekansı: {kesme_frekansi})")

        except Exception as e:
            print(f"Hata: Alçak geçiren filtre uygulanırken bir hata oluştu: {str(e)}")

    def yuksek_geciren_filtre_uygula(self):
        if self.working_image is None:
            print("Hata: İşlem yapılacak bir görüntü yok!")
            return

        try:
            kesme_frekansi = self.kesme_frekansi_scale.get()
            bgr_img = cv2.cvtColor(self.working_image, cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
            f_transform_shifted, _ = self.fft_goruntu(gray)

            if f_transform_shifted is None:
                print("Hata: Fourier dönüşümü hesaplanamadı!")
                return

            rows, cols = gray.shape
            mask = np.ones((rows, cols), np.uint8)
            center = (cols // 2, rows // 2)
            cv2.circle(mask, center, kesme_frekansi, 0, -1)
            filtered_fft = f_transform_shifted * mask
            filtered_image = np.fft.ifft2(np.fft.ifftshift(filtered_fft)).real
            filtered_image = cv2.normalize(filtered_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            self.working_image = cv2.cvtColor(filtered_image, cv2.COLOR_GRAY2RGB)
            self.goruntu_guncelle()
            print(f"Yüksek geçiren filtre uygulandı (Kesme frekansı: {kesme_frekansi})")

        except Exception as e:
            print(f"Hata: Yüksek geçiren filtre uygulanırken bir hata oluştu: {str(e)}")

    def gauss_alcak_geciren_uygula(self):
        if self.working_image is None:
            print("Hata: İşlem yapılacak bir görüntü yok!")
            return

        try:

            kesme_frekansi = self.kesme_frekansi_scale.get()
            bgr_img = cv2.cvtColor(self.working_image, cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
            f_transform_shifted, _ = self.fft_goruntu(gray)
            if f_transform_shifted is None:
                print("Hata: Fourier dönüşümü hesaplanamadı!")
                return

            rows, cols = gray.shape
            crow, ccol = rows // 2, cols // 2

            mask = np.zeros((rows, cols), np.float32)
            for i in range(rows):
                for j in range(cols):
                    # Merkeze olan uzaklık
                    d = np.sqrt((i - crow) ** 2 + (j - ccol) ** 2)
                    # Gaussian formülü H(u,v) = e^(-D^2/(2*D0^2))
                    mask[i, j] = np.exp(-(d ** 2) / (2 * (kesme_frekansi ** 2)))

            filtered_fft = f_transform_shifted * mask
            filtered_image = np.fft.ifft2(np.fft.ifftshift(filtered_fft)).real
            filtered_image = cv2.normalize(filtered_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            self.working_image = cv2.cvtColor(filtered_image, cv2.COLOR_GRAY2RGB)
            self.goruntu_guncelle()
            print(f"Gaussian alçak geçiren filtre uygulandı (Kesme frekansı: {kesme_frekansi})")

        except Exception as e:
            print(f"Hata: Gaussian alçak geçiren filtre uygulanırken bir hata oluştu: {str(e)}")

    def gauss_yuksek_geciren_uygula(self):
        if self.working_image is None:
            print("Hata: İşlem yapılacak bir görüntü yok!")
            return

        try:
            kesme_frekansi = self.kesme_frekansi_scale.get()
            bgr_img = cv2.cvtColor(self.working_image, cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
            f_transform_shifted, _ = self.fft_goruntu(gray)
            if f_transform_shifted is None:
                print("Hata: Fourier dönüşümü hesaplanamadı!")
                return

            rows, cols = gray.shape
            crow, ccol = rows // 2, cols // 2

            mask = np.zeros((rows, cols), np.float32)
            for i in range(rows):
                for j in range(cols):
                    # Merkeze olan uzaklık
                    d = np.sqrt((i - crow) ** 2 + (j - ccol) ** 2)
                    # Gaussian formülü H(u,v) = 1 - e^(-D^2/(2*D0^2))
                    mask[i, j] = 1 - np.exp(-(d ** 2) / (2 * (kesme_frekansi ** 2)))

            filtered_fft = f_transform_shifted * mask
            filtered_image = np.fft.ifft2(np.fft.ifftshift(filtered_fft)).real
            filtered_image = cv2.normalize(filtered_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            self.working_image = cv2.cvtColor(filtered_image, cv2.COLOR_GRAY2RGB)
            self.goruntu_guncelle()
            print(f"Gaussian yüksek geçiren filtre uygulandı (Kesme frekansı: {kesme_frekansi})")

        except Exception as e:
            print(f"Hata: Gaussian yüksek geçiren filtre uygulanırken bir hata oluştu: {str(e)}")

    def toggle_kenar_menu(self):

        if self.kenar_expand:
            self.kenar_menu_frame.pack_forget()
            self.kenar_btn.config(text="Kenar Bulma Algoritmaları ►")
            self.kenar_expand = False
        else:
            self.kenar_menu_frame.pack(fill=tk.X, padx=5, pady=2)
            self.kenar_btn.config(text="Kenar Bulma Algoritmaları ▼")
            self.kenar_expand = True

    def kenar_bulma_uygula(self, algoritma):

        if self.working_image is None:
            print("Hata: İşlem yapılacak bir görüntü yok!")
            return

        try:
            bgr_img = cv2.cvtColor(self.working_image, cv2.COLOR_RGB2BGR)
            if algoritma == "sobel":
                # Sobel kenar bulma
                sobel_x, sobel_y, sobel_magnitude = self.sobel_kenar_bulma(bgr_img)
                result_img = cv2.cvtColor(sobel_magnitude, cv2.COLOR_GRAY2RGB)

            elif algoritma == "prewitt":
                # Prewitt kenar bulma
                prewitt_x, prewitt_y, prewitt_magnitude = self.prewitt_kenar_bulma(bgr_img)
                result_img = cv2.cvtColor(prewitt_magnitude, cv2.COLOR_GRAY2RGB)

            elif algoritma == "roberts":
                # Roberts Cross kenar bulma
                roberts_x, roberts_y, roberts_magnitude = self.roberts_cross_kenar_bulma(bgr_img)
                result_img = cv2.cvtColor(roberts_magnitude, cv2.COLOR_GRAY2RGB)

            elif algoritma == "compass":
                # Compass kenar bulma
                compass_edges = self.compass_kenar_bulma(bgr_img)
                result_img = cv2.cvtColor(compass_edges, cv2.COLOR_GRAY2RGB)

            elif algoritma == "canny":
                # Canny kenar bulma
                canny_edges = self.canny_kenar_bulma(bgr_img)
                result_img = cv2.cvtColor(canny_edges, cv2.COLOR_GRAY2RGB)

            elif algoritma == "laplace":
                # Laplace kenar bulma
                laplacian = self.laplace_kenar_bulma(bgr_img)
                result_img = cv2.cvtColor(laplacian, cv2.COLOR_GRAY2RGB)

            else:
                print(f"Hata: Bilinmeyen kenar bulma algoritması: {algoritma}")
                return


            self.working_image = result_img
            self.goruntu_guncelle()

        except Exception as e:
            print(f"Hata: {algoritma} kenar bulma algoritması uygulanırken bir hata oluştu: {str(e)}")

    def sobel_kenar_bulma(self, img, ksize=3, normalize=True):
        if img is None:
            print("Hata: İşlenecek görüntü bulunamadı!")
            return None, None, None

        try:
            if len(img.shape) > 2:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                gray = img.copy()

            sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
            sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)
            sobel_magnitude = cv2.magnitude(sobel_x, sobel_y)

            if normalize:
                sobel_x = cv2.normalize(sobel_x, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                sobel_y = cv2.normalize(sobel_y, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                sobel_magnitude = cv2.normalize(sobel_magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

            return sobel_x, sobel_y, sobel_magnitude

        except Exception as e:
            print(f"Hata: Sobel kenar bulma algoritması uygulanırken bir hata oluştu: {str(e)}")
            return None, None, None

    def prewitt_kenar_bulma(self, img, normalize=True):
        if img is None:
            print("Hata: İşlenecek görüntü bulunamadı!")
            return None, None, None

        try:
            if len(img.shape) > 2:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                gray = img.copy()

            kernel_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
            kernel_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
            prewitt_x = cv2.filter2D(gray, cv2.CV_64F, kernel_x)
            prewitt_y = cv2.filter2D(gray, cv2.CV_64F, kernel_y)
            prewitt_magnitude = cv2.magnitude(prewitt_x, prewitt_y)

            if normalize:
                prewitt_x = cv2.normalize(prewitt_x, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                prewitt_y = cv2.normalize(prewitt_y, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                prewitt_magnitude = cv2.normalize(prewitt_magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

            return prewitt_x, prewitt_y, prewitt_magnitude

        except Exception as e:
            print(f"Hata: Prewitt kenar bulma algoritması uygulanırken bir hata oluştu: {str(e)}")
            return None, None, None

    def roberts_cross_kenar_bulma(self, img, normalize=True):

        if img is None:
            print("Hata: İşlenecek görüntü bulunamadı!")
            return None, None, None

        try:

            if len(img.shape) > 2:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                gray = img.copy()

            kernel_x = np.array([[1, 0], [0, -1]], dtype=np.float32)
            kernel_y = np.array([[0, 1], [-1, 0]], dtype=np.float32)
            roberts_x = cv2.filter2D(gray, cv2.CV_64F, kernel_x)
            roberts_y = cv2.filter2D(gray, cv2.CV_64F, kernel_y)
            roberts_magnitude = cv2.magnitude(roberts_x, roberts_y)

            if normalize:
                roberts_x = cv2.normalize(roberts_x, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                roberts_y = cv2.normalize(roberts_y, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                roberts_magnitude = cv2.normalize(roberts_magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

            return roberts_x, roberts_y, roberts_magnitude

        except Exception as e:
            print(f"Hata: Roberts Cross kenar bulma algoritması uygulanırken bir hata oluştu: {str(e)}")
            return None, None, None

    def compass_kenar_bulma(self, img, normalize=True):
        if img is None:
            print("Hata: İşlenecek görüntü bulunamadı!")
            return None

        try:

            if len(img.shape) > 2:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                gray = img.copy()

            compass_kernels = [
                np.array([[-1, -1, -1], [1, 1, 1], [1, 1, 1]]),  # Doğu (E)
                np.array([[1, 1, 1], [1, 1, 1], [-1, -1, -1]]),  # Batı (W)
                np.array([[-1, 1, 1], [-1, 1, 1], [-1, 1, 1]]),  # Kuzey (N)
                np.array([[1, 1, -1], [1, 1, -1], [1, 1, -1]]),  # Güney (S)
                np.array([[-1, -1, 1], [-1, 1, 1], [1, 1, 1]]),  # Kuzeydoğu (NE)
                np.array([[1, -1, -1], [1, 1, -1], [1, 1, 1]]),  # Kuzeybatı (NW)
                np.array([[1, 1, 1], [1, 1, -1], [1, -1, -1]]),  # Güneydoğu (SE)
                np.array([[1, 1, 1], [-1, 1, 1], [-1, -1, 1]])  # Güneybatı (SW)
            ]

            compass_edges = np.zeros_like(gray, dtype=np.float32)

            for kernel in compass_kernels:
                edge = cv2.filter2D(gray, cv2.CV_32F, kernel)
                compass_edges = np.maximum(compass_edges, edge)

            if normalize:
                compass_edges = cv2.normalize(compass_edges, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

            return compass_edges

        except Exception as e:
            print(f"Hata: Compass kenar bulma algoritması uygulanırken bir hata oluştu: {str(e)}")
            return None

    def canny_kenar_bulma(self, img, alt_esik=50, ust_esik=150, aperture_size=3):
        if img is None:
            print("Hata: İşlenecek görüntü bulunamadı!")
            return None

        try:

            if len(img.shape) > 2:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                gray = img.copy()
            canny_edges = cv2.Canny(gray, alt_esik, ust_esik, apertureSize=aperture_size)

            return canny_edges

        except Exception as e:
            print(f"Hata: Canny kenar bulma algoritması uygulanırken bir hata oluştu: {str(e)}")
            return None

    def laplace_kenar_bulma(self, img, ksize=3, normalize=True):

        if img is None:
            print("Hata: İşlenecek görüntü bulunamadı!")
            return None

        try:

            if len(img.shape) > 2:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                gray = img.copy()
            laplacian = cv2.Laplacian(gray, cv2.CV_64F, ksize=ksize)
            laplacian_abs = np.abs(laplacian)

            if normalize:
                laplacian_abs = cv2.normalize(laplacian_abs, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

            return laplacian_abs

        except Exception as e:
            print(f"Hata: Laplace kenar bulma algoritması uygulanırken bir hata oluştu: {str(e)}")
            return None

    def gabor_filtre_uygula(self):

        if self.working_image is None:
            print("Hata: İşlem yapılacak bir görüntü yok!")
            messagebox.showerror("Hata", "İşlem yapılacak bir görüntü yok!")
            return

        try:

            params_window = tk.Toplevel(self.root)
            params_window.title("Gabor Filtresi Parametreleri")
            params_window.geometry("400x400")  # Daha büyük pencere boyutu
            params_window.resizable(False, False)

            baslik_label = tk.Label(params_window, text="Gabor Filtresi Ayarları", font=("Arial", 12, "bold"))
            baslik_label.grid(row=0, column=0, columnspan=2, pady=10)

            tk.Label(params_window, text="Çekirdek Boyutu:").grid(row=1, column=0, sticky='w', padx=10, pady=5)
            kernel_size_var = tk.IntVar(value=21)
            kernel_size_scale = tk.Scale(params_window, from_=3, to=31, resolution=2, orient='horizontal',
                                         variable=kernel_size_var)
            kernel_size_scale.grid(row=1, column=1, sticky='we', padx=10, pady=5)

            tk.Label(params_window, text="Sigma (Standart Sapma):").grid(row=2, column=0, sticky='w', padx=10, pady=5)
            sigma_var = tk.DoubleVar(value=5.0)
            sigma_scale = tk.Scale(params_window, from_=1.0, to=10.0, resolution=0.1, orient='horizontal',
                                   variable=sigma_var)
            sigma_scale.grid(row=2, column=1, sticky='we', padx=10, pady=5)

            tk.Label(params_window, text="Theta (Yönelim):").grid(row=3, column=0, sticky='w', padx=10, pady=5)
            theta_var = tk.DoubleVar(value=45)  # derece cinsinden
            theta_scale = tk.Scale(params_window, from_=0, to=180, resolution=15, orient='horizontal',
                                   variable=theta_var)
            theta_scale.grid(row=3, column=1, sticky='we', padx=10, pady=5)

            tk.Label(params_window, text="Lambda (Dalga Boyu):").grid(row=4, column=0, sticky='w', padx=10, pady=5)
            lambda_var = tk.DoubleVar(value=10.0)
            lambda_scale = tk.Scale(params_window, from_=5.0, to=20.0, resolution=0.5, orient='horizontal',
                                    variable=lambda_var)
            lambda_scale.grid(row=4, column=1, sticky='we', padx=10, pady=5)

            tk.Label(params_window, text="Gamma (En-Boy Oranı):").grid(row=5, column=0, sticky='w', padx=10, pady=5)
            gamma_var = tk.DoubleVar(value=0.5)
            gamma_scale = tk.Scale(params_window, from_=0.1, to=1.0, resolution=0.1, orient='horizontal',
                                   variable=gamma_var)
            gamma_scale.grid(row=5, column=1, sticky='we', padx=10, pady=5)

            button_frame = tk.Frame(params_window)
            button_frame.grid(row=6, column=0, columnspan=2, pady=25)


            def apply_gabor():
                try:
                    print("Gabor filtresi uygulanıyor...")
                    kernel_size = kernel_size_var.get()
                    sigma = sigma_var.get()
                    theta = theta_var.get() * np.pi / 180.0  # Dereceyi radyana çevir
                    lambd = lambda_var.get()
                    gamma = gamma_var.get()

                    print(
                        f"Parametreler: kernel_size={kernel_size}, sigma={sigma}, theta={theta}, lambda={lambd}, gamma={gamma}")

                    bgr_img = cv2.cvtColor(self.working_image, cv2.COLOR_RGB2BGR)
                    print("Görüntü BGR formatına dönüştürüldü")


                    try:
                        gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)

                        kernel = cv2.getGaborKernel(
                            (kernel_size, kernel_size),
                            sigma,
                            theta,
                            lambd,
                            gamma,
                            0,
                            ktype=cv2.CV_32F
                        )

                        kernel /= np.sum(np.abs(kernel))

                        filtered_img = cv2.filter2D(gray, cv2.CV_8UC3, kernel)

                        filtered_img = cv2.normalize(filtered_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

                        result = cv2.cvtColor(filtered_img, cv2.COLOR_GRAY2BGR)
                        print("Gabor filtresi başarıyla uygulandı")

                    except Exception as e:
                        print(f"Gabor filtresi uygulanırken hata oluştu: {str(e)}")
                        # Global fonksiyonu kullanmayı dene
                        print("Global gabor_filtre fonksiyonu kullanılıyor...")
                        result = gabor_filtre(bgr_img, kernel_size, sigma, theta, lambd, gamma)

                        if result is None:
                            print("Gabor filtresi None döndürdü, işlem başarısız!")
                            messagebox.showerror("Hata", "Gabor filtresi uygulanamadı!")
                            return

                        result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)

                    result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
                    print("Sonuç RGB formatına dönüştürüldü")

                    self.working_image = result_rgb
                    self.goruntu_guncelle()
                    print("Görüntü güncellendi, işlem tamamlandı")

                    messagebox.showinfo("Bilgi", "Gabor filtresi başarıyla uygulandı!")

                    params_window.destroy()

                except Exception as e:
                    print(f"Hata: Gabor filtresi uygulanırken bir hata oluştu: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    messagebox.showerror("Hata", f"Gabor filtresi uygulanırken bir hata oluştu: {str(e)}")

            apply_btn = tk.Button(
                button_frame,
                text="FİLTREYİ UYGULA",
                command=apply_gabor,
                bg="#4CAF50",
                fg="white",
                font=("Arial", 12, "bold"),
                width=20,
                height=2
            )
            apply_btn.pack(pady=10)

            cancel_btn = tk.Button(
                button_frame,
                text="İptal",
                command=params_window.destroy,
                bg="#f44336",
                fg="white",
                width=15
            )
            cancel_btn.pack(pady=5)

            params_window.update_idletasks()
            width = params_window.winfo_width()
            height = params_window.winfo_height()
            x = (params_window.winfo_screenwidth() // 2) - (width // 2)
            y = (params_window.winfo_screenheight() // 2) - (height // 2)
            params_window.geometry('{}x{}+{}+{}'.format(width, height, x, y))

            params_window.transient(self.root)
            params_window.grab_set()
            self.root.wait_window(params_window)

        except Exception as e:
            print(f"Hata: Gabor filtresi penceresi açılırken bir hata oluştu: {str(e)}")
            import traceback
            traceback.print_exc()
            messagebox.showerror("Hata", f"Gabor filtresi penceresi açılırken bir hata oluştu: {str(e)}")

    def kmeans_uygula(self):
        if self.working_image is None:
            print("Hata: İşlem yapılacak bir görüntü yok!")
            return

        try:
            params_window = tk.Toplevel(self.root)
            params_window.title("K-Means Segmentasyon Parametreleri")
            params_window.geometry("300x200")
            params_window.resizable(False, False)

            tk.Label(params_window, text="Küme (K) Sayısı:").grid(row=0, column=0, sticky='w', padx=10, pady=5)
            k_var = tk.IntVar(value=3)
            k_scale = tk.Scale(params_window, from_=2, to=10, resolution=1, orient='horizontal', variable=k_var)
            k_scale.grid(row=0, column=1, sticky='we', padx=10, pady=5)

            tk.Label(params_window, text="Deneme Sayısı:").grid(row=1, column=0, sticky='w', padx=10, pady=5)
            attempts_var = tk.IntVar(value=10)
            attempts_scale = tk.Scale(params_window, from_=1, to=20, resolution=1, orient='horizontal',
                                      variable=attempts_var)
            attempts_scale.grid(row=1, column=1, sticky='we', padx=10, pady=5)

            def apply_kmeans():
                try:
                    k = k_var.get()
                    attempts = attempts_var.get()
                    bgr_img = cv2.cvtColor(self.working_image, cv2.COLOR_RGB2BGR)

                    try:

                        print("self.kmeans_segmentation fonksiyonu çağrılıyor...")
                        result = self.kmeans_segmentation(bgr_img, k=k, attempts=attempts)

                        if isinstance(result, tuple) and len(result) >= 2 and result[1] is not None:

                            result_img = result[1]
                            print("Segmentasyon başarılı oldu, segmente edilmiş görüntü alındı")
                        else:

                            print("Hata: kmeans_segmentation fonksiyonu beklenmeyen bir değer döndürdü")

                            print("Manuel K-Means segmentasyon uygulanıyor...")
                            result_img = self.kmeans_manuel(bgr_img, k, attempts)
                            if result_img is None:
                                messagebox.showerror("Hata", "K-Means segmentasyon başarısız oldu!")
                                return
                    except Exception as e:
                        print(f"kmeans_segmentation fonksiyonunda hata: {str(e)}")

                        print("Manuel K-Means segmentasyon uygulanıyor...")
                        result_img = self.kmeans_manuel(bgr_img, k, attempts)
                        if result_img is None:
                            import traceback
                            traceback.print_exc()
                            messagebox.showerror("Hata", f"K-Means segmentasyon başarısız oldu: {str(e)}")
                            return

                    result_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
                    print("Sonuç RGB formatına dönüştürüldü")

                    self.working_image = result_rgb
                    self.goruntu_guncelle()
                    print("Görüntü güncellendi, işlem tamamlandı")

                    params_window.destroy()

                except Exception as e:
                    print(f"Hata: K-Means segmentasyon uygulanırken bir hata oluştu: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    messagebox.showerror("Hata", f"K-Means segmentasyon uygulanırken bir hata oluştu: {str(e)}")

            apply_btn = tk.Button(params_window, text="Uygula", command=apply_kmeans, bg="#4CAF50", fg="white")
            apply_btn.grid(row=2, column=0, columnspan=2, pady=20)

            params_window.update_idletasks()
            width = params_window.winfo_width()
            height = params_window.winfo_height()
            x = (self.root.winfo_screenwidth() // 2) - (width // 2)
            y = (self.root.winfo_screenheight() // 2) - (height // 2)
            params_window.geometry('{}x{}+{}+{}'.format(width, height, x, y))

        except Exception as e:
            print(f"Hata: K-Means segmentasyon penceresi açılırken bir hata oluştu: {str(e)}")
            import traceback
            traceback.print_exc()
            messagebox.showerror("Hata", f"K-Means segmentasyon penceresi açılırken bir hata oluştu: {str(e)}")

    def kmeans_manuel(self, img, k=3, attempts=10):
        try:
            if img is None:
                print("Hata: İşlenecek görüntü bulunamadı!")
                return None

            pixel_values = img.reshape((-1, 3))
            pixel_values = np.float32(pixel_values)
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
            _, labels, centers = cv2.kmeans(pixel_values, k, None, criteria, attempts, cv2.KMEANS_RANDOM_CENTERS)
            centers = np.uint8(centers)
            segmented_data = centers[labels.flatten()]
            segmented_image = segmented_data.reshape(img.shape)
            return segmented_image

        except Exception as e:
            print(f"Hata: Manuel K-Means segmentasyon işlemi uygulanırken bir hata oluştu: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

    def kmeans_segmentation(self, img, k=3, attempts=10):
        try:

            try:
                from Hafta6Ogrendiklerimiz import kmeans_segmentation
                print("Hafta6Ogrendiklerimiz modülünden kmeans_segmentation import edildi.")
                return kmeans_segmentation(img, k, attempts)
            except ImportError:
                segmented_image = self.kmeans_manuel(img, k, attempts)
                return img, segmented_image, None

        except Exception as e:
            print(f"Hata: K-Means segmentasyon işlemi uygulanırken bir hata oluştu: {str(e)}")
            import traceback
            traceback.print_exc()
            return None, None, None

    def erode_uygula(self):
        if self.working_image is None:
            print("Hata: İşlem yapılacak bir görüntü yok!")
            return

        try:
            kernel_size = self.kernel_boyut_scale.get()
            kernel_shape = self.kernel_sekil_var.get()
            iterations = self.iterasyon_scale.get()

            bgr_img = cv2.cvtColor(self.working_image, cv2.COLOR_RGB2BGR)

            result = erode_islem(bgr_img, kernel_size, kernel_shape, iterations)

            if result is not None:
                result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

                self.working_image = result_rgb
                self.goruntu_guncelle()

                print(
                    f"Aşındırma işlemi başarıyla uygulandı (Boyut: {kernel_size}x{kernel_size}, Şekil: {kernel_shape}, İterasyon: {iterations})")

        except Exception as e:
            print(f"Hata: Aşındırma işlemi uygulanırken bir hata oluştu: {str(e)}")
            import traceback
            traceback.print_exc()

    def dilate_uygula(self):
        if self.working_image is None:
            print("Hata: İşlem yapılacak bir görüntü yok!")
            return

        try:
            kernel_size = self.kernel_boyut_scale.get()
            kernel_shape = self.kernel_sekil_var.get()
            iterations = self.iterasyon_scale.get()

            bgr_img = cv2.cvtColor(self.working_image, cv2.COLOR_RGB2BGR)

            result = dilate_islem(bgr_img, kernel_size, kernel_shape, iterations)

            if result is not None:
                result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

                self.working_image = result_rgb
                self.goruntu_guncelle()

        except Exception as e:
            print(f"Hata: Genişletme işlemi uygulanırken bir hata oluştu: {str(e)}")

    def opening_uygula(self):
        if self.working_image is None:
            print("Hata: İşlem yapılacak bir görüntü yok!")
            return

        try:
            kernel_size = self.kernel_boyut_scale.get()
            kernel_shape = self.kernel_sekil_var.get()
            bgr_img = cv2.cvtColor(self.working_image, cv2.COLOR_RGB2BGR)
            result = opening_islem(bgr_img, kernel_size, kernel_shape)

            if result is not None:
                result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

                self.working_image = result_rgb
                self.goruntu_guncelle()

        except Exception as e:
            print(f"Hata: Açma işlemi uygulanırken bir hata oluştu: {str(e)}")

    def closing_uygula(self):
        if self.working_image is None:
            print("Hata: İşlem yapılacak bir görüntü yok!")
            return

        try:
            kernel_size = self.kernel_boyut_scale.get()
            kernel_shape = self.kernel_sekil_var.get()

            bgr_img = cv2.cvtColor(self.working_image, cv2.COLOR_RGB2BGR)

            result = closing_islem(bgr_img, kernel_size, kernel_shape)

            if result is not None:
                result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

                self.working_image = result_rgb
                self.goruntu_guncelle()

        except Exception as e:
            print(f"Hata: Kapama işlemi uygulanırken bir hata oluştu: {str(e)}")

    def kernel_goster_uygula(self):
        try:
            kernel_size = self.kernel_boyut_scale.get()
            kernel_shape = self.kernel_sekil_var.get()

            kernel_goster(kernel_size, kernel_shape)
        except Exception as e:
            print(f"Hata: Yapısal elemanlar gösterilirken bir hata oluştu: {str(e)}")

    def toggle_morfolojik_menu(self):
        if self.morfolojik_expand:
            self.morfolojik_scrollable_frame.pack_forget()
            self.morfolojik_canvas.pack_forget()
            self.morfolojik_scrollbar.pack_forget()
            self.morfolojik_btn.config(text="Morfolojik İşlemler ►")
            self.morfolojik_expand = False

            self.morfolojik_canvas.unbind_all("<MouseWheel>")
            self.sol_canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        else:
            self.morfolojik_scrollable_frame.pack(fill=tk.X, padx=5, pady=2, expand=True)
            self.morfolojik_btn.config(text="Morfolojik İşlemler ▼")
            self.morfolojik_expand = True

            self.morfolojik_canvas.update_idletasks()
            visible_height = min(250, self.morfolojik_menu_frame.winfo_reqheight())
            self.morfolojik_canvas.config(height=visible_height)
            self.morfolojik_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
            self.morfolojik_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

            self.morfolojik_canvas.bind_all("<MouseWheel>", self._on_morfolojik_mousewheel)

            morfolojik_y_pos = self.morfolojik_islemler_frame.winfo_y()
            self.sol_canvas.yview_moveto(morfolojik_y_pos / self.sol_icerik_frame.winfo_height())

    def _on_morfolojik_mousewheel(self, event):
        if self.morfolojik_expand:
            self.morfolojik_canvas.yview_scroll(int(-3 * (event.delta / 120)), "units")

    def kernel_boyutu_degistir(self, val):
        try:
            deger = int(float(val))
            if deger % 2 == 0:
                deger += 1
                self.kernel_boyut_scale.set(deger)
            self.kernel_boyut_label.config(text=str(deger))
        except Exception as e:
            print(f"Hata: Yapısal eleman boyutu değiştirilirken bir hata oluştu: {str(e)}")

    def iterasyon_degistir(self, val):
        try:
            deger = int(float(val))
            self.iterasyon_label.config(text=str(deger))
        except Exception as e:
            print(f"Hata: İterasyon sayısı değiştirilirken bir hata oluştu: {str(e)}")

    def kalinlik_degistir(self, val):
        try:
            deger = int(float(val))
            self.kalinlik_deger_label.config(text=str(deger))
            self.kalinlik_scale_value.set(deger)
        except Exception as e:
            print(f"Hata: Çizgi kalınlığı değiştirilirken bir hata oluştu: {str(e)}")

    def hough_dogru_uygula(self):
        if self.working_image is None:
            print("Hata: İşlem yapılacak bir görüntü yok!")
            messagebox.showerror("Hata", "İşlem yapılacak bir görüntü yok!")
            return

        try:
            params_window = tk.Toplevel(self.root)
            params_window.title("Hough Doğru Tespiti Parametreleri")
            params_window.geometry("400x400")
            params_window.resizable(False, False)

            baslik_label = tk.Label(params_window, text="Hough Doğru Tespiti Ayarları", font=("Arial", 12, "bold"))
            baslik_label.grid(row=0, column=0, columnspan=2, pady=10)

            tk.Label(params_window, text="Eşik Değeri:").grid(row=1, column=0, sticky='w', padx=10, pady=5)
            threshold_var = tk.IntVar(value=150)
            threshold_scale = tk.Scale(params_window, from_=10, to=300, resolution=10, orient='horizontal',
                                       variable=threshold_var)
            threshold_scale.grid(row=1, column=1, sticky='we', padx=10, pady=5)

            tk.Label(params_window, text="Min. Çizgi Uzunluğu:").grid(row=2, column=0, sticky='w', padx=10, pady=5)
            min_line_length_var = tk.IntVar(value=50)
            min_line_length_scale = tk.Scale(params_window, from_=10, to=200, resolution=5, orient='horizontal',
                                             variable=min_line_length_var)
            min_line_length_scale.grid(row=2, column=1, sticky='we', padx=10, pady=5)

            tk.Label(params_window, text="Maks. Boşluk:").grid(row=3, column=0, sticky='w', padx=10, pady=5)
            max_gap_var = tk.IntVar(value=10)
            max_gap_scale = tk.Scale(params_window, from_=1, to=50, resolution=1, orient='horizontal',
                                     variable=max_gap_var)
            max_gap_scale.grid(row=3, column=1, sticky='we', padx=10, pady=5)

            tk.Label(params_window, text="Kenar Alt Eşik:").grid(row=4, column=0, sticky='w', padx=10, pady=5)
            canny_low_var = tk.IntVar(value=50)
            canny_low_scale = tk.Scale(params_window, from_=10, to=200, resolution=5, orient='horizontal',
                                       variable=canny_low_var)
            canny_low_scale.grid(row=4, column=1, sticky='we', padx=10, pady=5)

            tk.Label(params_window, text="Kenar Üst Eşik:").grid(row=5, column=0, sticky='w', padx=10, pady=5)
            canny_high_var = tk.IntVar(value=150)
            canny_high_scale = tk.Scale(params_window, from_=50, to=300, resolution=5, orient='horizontal',
                                        variable=canny_high_var)
            canny_high_scale.grid(row=5, column=1, sticky='we', padx=10, pady=5)

            button_frame = tk.Frame(params_window)
            button_frame.grid(row=6, column=0, columnspan=2, pady=25)

            def apply_hough_lines():
                print("Hough doğruları tespiti başlatılıyor...")
                try:
                    threshold = threshold_var.get()
                    min_line_length = min_line_length_var.get()
                    max_gap = max_gap_var.get()
                    canny_low = canny_low_var.get()
                    canny_high = canny_high_var.get()

                    print(f"Parametreler: threshold={threshold}, min_line_length={min_line_length}, max_gap={max_gap}")
                    print(f"Canny parametreleri: alt_esik={canny_low}, ust_esik={canny_high}")

                    bgr_img = cv2.cvtColor(self.working_image, cv2.COLOR_RGB2BGR)
                    print(f"Girdi görüntü boyutları: {bgr_img.shape}")

                    print("Manuel Hough doğruları tespiti yapılıyor...")
                    result_img = bgr_img.copy()

                    gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
                    print(f"Gri tonlamalı görüntü boyutları: {gray.shape}")

                    edges = cv2.Canny(gray, canny_low, canny_high, apertureSize=3)
                    print(f"Kenar görüntüsü boyutları: {edges.shape}")

                    try:

                        print(
                            f"HoughLinesP parametreleri: edges.shape={edges.shape}, rho=1, theta=np.pi/180, threshold={threshold}, minLineLength={min_line_length}, maxLineGap={max_gap}")
                        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold,
                                                minLineLength=min_line_length,
                                                maxLineGap=max_gap)

                        print(f"Tespit edilen çizgiler: {lines}")

                        if lines is not None:
                            print(f"Tespit edilen çizgi sayısı: {len(lines)}")
                            for i, line in enumerate(lines):
                                if i < 5:
                                    print(f"Çizgi {i}: {line[0]}")
                                x1, y1, x2, y2 = line[0]
                                cv2.line(result_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        else:
                            print("Çizgi tespit edilemedi!")
                            messagebox.showinfo("Bilgi",
                                                "Hough doğru tespiti sonucu: Çizgi bulunamadı.\n\nYeni parametreler deneyebilirsiniz.")
                            return
                    except Exception as hough_error:
                        print(f"HoughLinesP çağrısında hata: {hough_error}")
                        import traceback
                        traceback.print_exc()
                        messagebox.showerror("Hata", f"Hough doğru tespiti sırasında bir hata oluştu: {hough_error}")
                        return

                    if result_img is not None:

                        result_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
                        print("Sonuç RGB formatına dönüştürüldü")

                        self.working_image = result_rgb
                        self.goruntu_guncelle()
                        print("Görüntü güncellendi, işlem tamamlandı")

                        if lines is not None and len(lines) > 0:
                            messagebox.showinfo("Bilgi",
                                                f"Hough doğru tespiti başarılı: {len(lines)} çizgi tespit edildi.")


                        params_window.destroy()
                    else:
                        print("Hata: Hough doğru tespiti sonucu alınamadı.")
                        messagebox.showerror("Hata", "Hough doğru tespiti sonucu alınamadı.")

                except Exception as e:
                    print(f"Hata: Hough doğru tespiti uygulanırken bir hata oluştu: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    messagebox.showerror("Hata", f"Hough doğru tespiti uygulanırken bir hata oluştu: {str(e)}")

            apply_btn = tk.Button(
                button_frame,
                text="DOĞRULARI TESPIT ET",
                command=apply_hough_lines,
                bg="#4CAF50",
                fg="white",
                font=("Arial", 12, "bold"),
                width=20,
                height=2
            )
            apply_btn.pack(pady=10)

            cancel_btn = tk.Button(
                button_frame,
                text="İptal",
                command=params_window.destroy,
                bg="#f44336",
                fg="white",
                width=15
            )
            cancel_btn.pack(pady=5)

            params_window.update_idletasks()
            width = params_window.winfo_width()
            height = params_window.winfo_height()
            x = (params_window.winfo_screenwidth() // 2) - (width // 2)
            y = (params_window.winfo_screenheight() // 2) - (height // 2)
            params_window.geometry('{}x{}+{}+{}'.format(width, height, x, y))

            params_window.transient(self.root)
            params_window.grab_set()
            self.root.wait_window(params_window)

        except Exception as e:
            print(f"Hata: Hough doğru tespiti penceresi açılırken bir hata oluştu: {str(e)}")
            import traceback
            traceback.print_exc()
            messagebox.showerror("Hata", f"Hough doğru tespiti penceresi açılırken bir hata oluştu: {str(e)}")

    def hough_cember_uygula(self):
        if self.working_image is None:
            print("Hata: İşlem yapılacak bir görüntü yok!")
            messagebox.showerror("Hata", "İşlem yapılacak bir görüntü yok!")
            return

        try:

            params_window = tk.Toplevel(self.root)
            params_window.title("Hough Çember Tespiti Parametreleri")
            params_window.geometry("400x450")
            params_window.resizable(False, False)

            baslik_label = tk.Label(params_window, text="Hough Çember Tespiti Ayarları", font=("Arial", 12, "bold"))
            baslik_label.grid(row=0, column=0, columnspan=2, pady=10)

            tk.Label(params_window, text="Min. Çember Mesafesi:").grid(row=1, column=0, sticky='w', padx=10, pady=5)
            min_dist_var = tk.IntVar(value=30)
            min_dist_scale = tk.Scale(params_window, from_=10, to=100, resolution=5, orient='horizontal',
                                      variable=min_dist_var)
            min_dist_scale.grid(row=1, column=1, sticky='we', padx=10, pady=5)

            tk.Label(params_window, text="Param1 (Canny Üst Eşik):").grid(row=2, column=0, sticky='w', padx=10, pady=5)
            param1_var = tk.IntVar(value=50)
            param1_scale = tk.Scale(params_window, from_=10, to=300, resolution=10, orient='horizontal',
                                    variable=param1_var)
            param1_scale.grid(row=2, column=1, sticky='we', padx=10, pady=5)

            tk.Label(params_window, text="Param2 (Merkez Eşik):").grid(row=3, column=0, sticky='w', padx=10, pady=5)
            param2_var = tk.IntVar(value=30)
            param2_scale = tk.Scale(params_window, from_=5, to=100, resolution=5, orient='horizontal',
                                    variable=param2_var)
            param2_scale.grid(row=3, column=1, sticky='we', padx=10, pady=5)

            tk.Label(params_window, text="Min. Yarıçap:").grid(row=4, column=0, sticky='w', padx=10, pady=5)
            min_radius_var = tk.IntVar(value=10)
            min_radius_scale = tk.Scale(params_window, from_=1, to=100, resolution=1, orient='horizontal',
                                        variable=min_radius_var)
            min_radius_scale.grid(row=4, column=1, sticky='we', padx=10, pady=5)

            tk.Label(params_window, text="Maks. Yarıçap:").grid(row=5, column=0, sticky='w', padx=10, pady=5)
            max_radius_var = tk.IntVar(value=100)
            max_radius_scale = tk.Scale(params_window, from_=10, to=300, resolution=10, orient='horizontal',
                                        variable=max_radius_var)
            max_radius_scale.grid(row=5, column=1, sticky='we', padx=10, pady=5)
            button_frame = tk.Frame(params_window)
            button_frame.grid(row=6, column=0, columnspan=2, pady=25)

            def apply_hough_circles():
                print("Hough çemberleri tespiti başlatılıyor...")
                try:
                    min_dist = min_dist_var.get()
                    param1 = param1_var.get()
                    param2 = param2_var.get()
                    min_radius = min_radius_var.get()
                    max_radius = max_radius_var.get()

                    print(
                        f"Parametreler: min_dist={min_dist}, param1={param1}, param2={param2}, min_radius={min_radius}, max_radius={max_radius}")
                    bgr_img = cv2.cvtColor(self.working_image, cv2.COLOR_RGB2BGR)
                    print(f"Girdi görüntü boyutları: {bgr_img.shape}")
                    print("Manuel Hough çemberleri tespiti yapılıyor...")
                    result_img = bgr_img.copy()
                    gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
                    print(f"Gri tonlamalı görüntü boyutları: {gray.shape}")
                    blurred = cv2.GaussianBlur(gray, (9, 9), 2)

                    try:

                        print(
                            f"HoughCircles parametreleri: blurred.shape={blurred.shape}, dp=1.0, min_dist={min_dist}, param1={param1}, param2={param2}, minRadius={min_radius}, maxRadius={max_radius}")
                        circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1.0, min_dist,
                                                   param1=param1, param2=param2,
                                                   minRadius=min_radius, maxRadius=max_radius)

                        print(f"Tespit edilen çemberler: {circles}")

                        if circles is not None:
                            circles = np.uint16(np.around(circles))
                            print(f"Tespit edilen çember sayısı: {len(circles[0])}")
                            for i, circle in enumerate(circles[0, :]):
                                if i < 5:
                                    print(f"Çember {i}: Merkez=({circle[0]}, {circle[1]}), Yarıçap={circle[2]}")
                                cv2.circle(result_img, (circle[0], circle[1]), circle[2], (0, 255, 0), 2)
                                cv2.circle(result_img, (circle[0], circle[1]), 2, (0, 0, 255), 3)
                        else:
                            print("Çember tespit edilemedi!")
                            messagebox.showinfo("Bilgi",
                                                "Hough çember tespiti sonucu: Çember bulunamadı.\n\nYeni parametreler deneyebilirsiniz.")
                            return
                    except Exception as hough_error:
                        print(f"HoughCircles çağrısında hata: {hough_error}")
                        import traceback
                        traceback.print_exc()
                        messagebox.showerror("Hata", f"Hough çember tespiti sırasında bir hata oluştu: {hough_error}")
                        return

                    if result_img is not None:

                        result_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
                        print("Sonuç RGB formatına dönüştürüldü")
                        self.working_image = result_rgb
                        self.goruntu_guncelle()
                        print("Görüntü güncellendi, işlem tamamlandı")

                        if circles is not None:
                            messagebox.showinfo("Bilgi",
                                                f"Hough çember tespiti başarılı: {len(circles[0])} çember tespit edildi.")

                        params_window.destroy()
                    else:
                        print("Hata: Hough çember tespiti sonucu alınamadı.")
                        messagebox.showerror("Hata", "Hough çember tespiti sonucu alınamadı.")

                except Exception as e:
                    print(f"Hata: Hough çember tespiti uygulanırken bir hata oluştu: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    messagebox.showerror("Hata", f"Hough çember tespiti uygulanırken bir hata oluştu: {str(e)}")

            apply_btn = tk.Button(
                button_frame,
                text="ÇEMBERLERİ TESPİT ET",
                command=apply_hough_circles,
                bg="#4CAF50",
                fg="white",
                font=("Arial", 12, "bold"),
                width=20,
                height=2
            )
            apply_btn.pack(pady=10)

            cancel_btn = tk.Button(
                button_frame,
                text="İptal",
                command=params_window.destroy,
                bg="#f44336",
                fg="white",
                width=15
            )
            cancel_btn.pack(pady=5)

            params_window.update_idletasks()
            width = params_window.winfo_width()
            height = params_window.winfo_height()
            x = (params_window.winfo_screenwidth() // 2) - (width // 2)
            y = (params_window.winfo_screenheight() // 2) - (height // 2)
            params_window.geometry('{}x{}+{}+{}'.format(width, height, x, y))

            params_window.transient(self.root)
            params_window.grab_set()
            self.root.wait_window(params_window)

        except Exception as e:
            print(f"Hata: Hough çember tespiti penceresi açılırken bir hata oluştu: {str(e)}")
            import traceback
            traceback.print_exc()
            messagebox.showerror("Hata", f"Hough çember tespiti penceresi açılırken bir hata oluştu: {str(e)}")

    def aynalama_popup(self):
        if self.working_image is None:
            messagebox.showerror("Hata", "İşlem yapılacak bir görüntü yok!")
            return

        popup = tk.Menu(self.root, tearoff=0)
        popup.add_command(label="Dikey Aynalama", command=self.aynalama_dikey_uygula)
        popup.add_command(label="Yatay Aynalama", command=self.aynalama_yatay_uygula)
        popup.add_command(label="Dikey ve Yatay Aynalama", command=self.aynalama_her_iki_uygula)

        try:
            x = self.aynalama_btn.winfo_rootx()
            y = self.aynalama_btn.winfo_rooty() + self.aynalama_btn.winfo_height()

            popup.tk_popup(x, y, 0)
        finally:
            popup.grab_release()

    def iterasyon_degistir(self, val):
        try:
            deger = int(float(val))
            self.iterasyon_deger_label.config(text=str(deger))
        except Exception as e:
            print(f"Hata: İterasyon değeri değiştirilirken bir hata oluştu: {str(e)}")

    def kalinlik_degistir(self, val):
        try:
            deger = int(float(val))
            self.kalinlik_deger_label.config(text=str(deger))
            self.kalinlik_scale_value.set(deger)
        except Exception as e:
            print(f"Hata: Çizgi kalınlığı değiştirilirken bir hata oluştu: {str(e)}")

    def on_mouse_down(self, event):
        print(f"Mouse Down - Drawing Mode: {self.drawing_mode}")
        if self.working_image is None or self.drawing_mode is None:
            print("Mouse Down - İşlem yapılamadı: Görüntü yok veya çizim modu aktif değil")
            return

        x, y = event.x, event.y
        print(f"Mouse Down - Koordinat: ({x}, {y})")

        img_height, img_width = self.working_image.shape[:2]
        panel_width = self.goruntu_label.winfo_width()
        panel_height = self.goruntu_label.winfo_height()

        scale = min(panel_width / img_width, panel_height / img_height)

        scaled_width = int(img_width * scale)
        scaled_height = int(img_height * scale)

        start_x = (panel_width - scaled_width) // 2
        start_y = (panel_height - scaled_height) // 2

        img_x = int((x - start_x) / scale)
        img_y = int((y - start_y) / scale)
        print(f"Mouse Down - Görüntü Koordinatı: ({img_x}, {img_y})")

        if 0 <= img_x < img_width and 0 <= img_y < img_height:
            self.start_point = (img_x, img_y)
            self.temp_image = self.working_image.copy()
            print(f"Mouse Down - Başlangıç noktası belirlendi: {self.start_point}")
        else:
            print("Mouse Down - Koordinat görüntü dışında")

    def on_mouse_move(self, event):
        if self.working_image is None or self.drawing_mode is None or self.start_point is None:
            return

        x, y = event.x, event.y
        print(f"Mouse Move - Koordinat: ({x}, {y}), Mod: {self.drawing_mode}")

        img_height, img_width = self.working_image.shape[:2]
        panel_width = self.goruntu_label.winfo_width()
        panel_height = self.goruntu_label.winfo_height()

        scale = min(panel_width / img_width, panel_height / img_height)

        scaled_width = int(img_width * scale)
        scaled_height = int(img_height * scale)

        start_x = (panel_width - scaled_width) // 2
        start_y = (panel_height - scaled_height) // 2

        img_x = int((x - start_x) / scale)
        img_y = int((y - start_y) / scale)

        img_x = max(0, min(img_x, img_width - 1))
        img_y = max(0, min(img_y, img_height - 1))

        try:
            renk = self.cizim_rengi.get()
            if renk == "red":
                color = (255, 0, 0)  # RGB format
            elif renk == "green":
                color = (0, 255, 0)
            elif renk == "blue":
                color = (0, 0, 255)
            else:
                color = (255, 0, 0)  # Default red
        except:
            print("Hata: Renk değişkeni bulunamadı, varsayılan olarak kırmızı kullanılıyor.")
            color = (255, 0, 0)

        try:
            kalinlik = self.kalinlik_scale_value.get()
            print(f"Mouse Move - Çizim: Renk={renk}, Kalınlık={kalinlik}")
        except:
            print("Hata: Kalınlık değişkeni bulunamadı, varsayılan olarak 2 kullanılıyor.")
            kalinlik = 2

        temp_draw = self.temp_image.copy()

        if self.drawing_mode == "cizgi":
            cv2.line(temp_draw, self.start_point, (img_x, img_y), color, kalinlik)
            print(f"Mouse Move - Çizgi çiziliyor: ({self.start_point}) -> ({img_x}, {img_y})")
        elif self.drawing_mode == "dikdortgen":
            cv2.rectangle(temp_draw, self.start_point, (img_x, img_y), color, kalinlik)
            print(f"Mouse Move - Dikdörtgen çiziliyor: ({self.start_point}) -> ({img_x}, {img_y})")
        elif self.drawing_mode == "daire":
            # Merkez ve yarıçap hesapla
            center_x = (self.start_point[0] + img_x) // 2
            center_y = (self.start_point[1] + img_y) // 2
            radius = int(((self.start_point[0] - img_x) ** 2 + (self.start_point[1] - img_y) ** 2) ** 0.5 // 2)
            cv2.circle(temp_draw, (center_x, center_y), radius, color, kalinlik)
            print(f"Mouse Move - Daire çiziliyor: Merkez=({center_x}, {center_y}), Yarıçap={radius}")

        self.working_image = temp_draw
        self.goruntu_guncelle()

    def on_mouse_up(self, event):
        print(f"Mouse Up - Drawing Mode: {self.drawing_mode}")
        if self.working_image is None or self.drawing_mode is None or self.start_point is None:
            print(
                "Mouse Up - İşlem yapılamadı: Görüntü yok, çizim modu aktif değil veya başlangıç noktası belirlenmemiş")
            return

        x, y = event.x, event.y
        print(f"Mouse Up - Koordinat: ({x}, {y})")

        img_height, img_width = self.working_image.shape[:2]
        panel_width = self.goruntu_label.winfo_width()
        panel_height = self.goruntu_label.winfo_height()

        scale = min(panel_width / img_width, panel_height / img_height)

        scaled_width = int(img_width * scale)
        scaled_height = int(img_height * scale)

        start_x = (panel_width - scaled_width) // 2
        start_y = (panel_height - scaled_height) // 2

        img_x = int((x - start_x) / scale)
        img_y = int((y - start_y) / scale)

        img_x = max(0, min(img_x, img_width - 1))
        img_y = max(0, min(img_y, img_height - 1))
        print(f"Mouse Up - Görüntü Koordinatı: ({img_x}, {img_y})")

        try:
            renk = self.cizim_rengi.get()
            if renk == "red":
                color = (255, 0, 0)
            elif renk == "green":
                color = (0, 255, 0)
            elif renk == "blue":
                color = (0, 0, 255)
            else:
                color = (255, 0, 0)
        except:

            print("Hata: Renk değişkeni bulunamadı, varsayılan olarak kırmızı kullanılıyor.")
            color = (255, 0, 0)

        try:
            kalinlik = self.kalinlik_scale_value.get()
        except:
            print("Hata: Kalınlık değişkeni bulunamadı, varsayılan olarak 2 kullanılıyor.")
            kalinlik = 2

        if self.drawing_mode == "cizgi":
            cv2.line(self.working_image, self.start_point, (img_x, img_y), color, kalinlik)
            print(f"Mouse Up - Çizgi çizildi: ({self.start_point}) -> ({img_x}, {img_y})")
        elif self.drawing_mode == "dikdortgen":
            cv2.rectangle(self.working_image, self.start_point, (img_x, img_y), color, kalinlik)
            print(f"Mouse Up - Dikdörtgen çizildi: ({self.start_point}) -> ({img_x}, {img_y})")
        elif self.drawing_mode == "daire":

            center_x = (self.start_point[0] + img_x) // 2
            center_y = (self.start_point[1] + img_y) // 2
            radius = int(((self.start_point[0] - img_x) ** 2 + (self.start_point[1] - img_y) ** 2) ** 0.5 // 2)
            cv2.circle(self.working_image, (center_x, center_y), radius, color, kalinlik)
            print(f"Mouse Up - Daire çizildi: Merkez=({center_x}, {center_y}), Yarıçap={radius}")

        self.start_point = None
        self.temp_image = None

        self.drawing_mode = None
        print("Mouse Up - Çizim tamamlandı, çizim modu sıfırlandı")

        self.goruntu_guncelle()

    def cizgi_ciz_uygula(self):
        if self.working_image is None:
            print("Çizgi Çiz - Hata: İşlem yapılacak bir görüntü yok!")
            messagebox.showerror("Hata", "İşlem yapılacak bir görüntü yok! Lütfen önce bir görüntü yükleyin.")
            return

        try:

            bgr_img = cv2.cvtColor(self.working_image, cv2.COLOR_RGB2BGR)
            cizim_goruntu = bgr_img.copy()
            pencere_adi = "Çizgi Çizme - İki nokta seçin (İptal: ESC, Tamamla: ENTER)"
            print("Not: Çizgi çizme işlemi için görüntü üzerinde 2 nokta seçmeniz gerekecek.")
            print("OpenCV penceresi açılacak ve fare ile nokta seçimi yapabileceksiniz.")
            print("İşlemi tamamlamak için ENTER tuşuna, iptal etmek için ESC tuşuna basın.")

            secilen_noktalar = []

            def nokta_sec(event, x, y, flags, param):
                nonlocal secilen_noktalar, cizim_goruntu

                if event == cv2.EVENT_LBUTTONDOWN:

                    if len(secilen_noktalar) < 2:
                        secilen_noktalar.append((x, y))
                        print(f"Nokta {len(secilen_noktalar)} Seçildi: {x}, {y}")

                        cizim_goruntu = bgr_img.copy()
                        for i, (px, py) in enumerate(secilen_noktalar):
                            cv2.circle(cizim_goruntu, (px, py), 5, (0, 0, 255), -1)
                            cv2.putText(cizim_goruntu, f"{i + 1}", (px + 10, py + 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                        if len(secilen_noktalar) == 2:
                            try:
                                renk = self.cizim_rengi.get()
                                if renk == "red":
                                    color = (0, 0, 255)
                                elif renk == "green":
                                    color = (0, 255, 0)
                                elif renk == "blue":
                                    color = (255, 0, 0)
                                else:
                                    color = (0, 0, 255)
                            except:
                                color = (0, 0, 255)

                            try:
                                kalinlik = self.kalinlik_scale_value.get()
                            except:
                                kalinlik = 2

                            pt1 = secilen_noktalar[0]
                            pt2 = secilen_noktalar[1]
                            cv2.line(cizim_goruntu, pt1, pt2, color, kalinlik)

                        cv2.imshow(pencere_adi, cizim_goruntu)

            cv2.namedWindow(pencere_adi)
            cv2.setMouseCallback(pencere_adi, nokta_sec)

            cv2.imshow(pencere_adi, bgr_img)

            while True:
                key = cv2.waitKey(0) & 0xFF

                # ESC ile iptal
                if key == 27:  # ESC
                    print("Çizgi çizme işlemi iptal edildi.")
                    cv2.destroyWindow(pencere_adi)
                    return

                # ENTER ile tamamla
                elif key == 13:  # ENTER
                    if len(secilen_noktalar) == 2:

                        self.working_image = cv2.cvtColor(cizim_goruntu, cv2.COLOR_BGR2RGB)
                        self.goruntu_guncelle()
                        print("Çizgi çizme işlemi tamamlandı.")
                        break
                    else:
                        print(f"Lütfen 2 nokta seçin! Şu ana kadar {len(secilen_noktalar)} nokta seçildi.")

            cv2.destroyWindow(pencere_adi)

        except Exception as e:
            print(f"Hata: Çizgi çizme işlemi sırasında bir hata oluştu: {str(e)}")
            traceback.print_exc()
            cv2.destroyAllWindows()

    def dikdortgen_ciz_uygula(self):
        if self.working_image is None:
            print("Dikdörtgen Çiz - Hata: İşlem yapılacak bir görüntü yok!")
            messagebox.showerror("Hata", "İşlem yapılacak bir görüntü yok! Lütfen önce bir görüntü yükleyin.")
            return

        try:
            bgr_img = cv2.cvtColor(self.working_image, cv2.COLOR_RGB2BGR)
            cizim_goruntu = bgr_img.copy()

            pencere_adi = "Dikdörtgen Çizme - İki nokta seçin (İptal: ESC, Tamamla: ENTER)"

            print("Not: Dikdörtgen çizme işlemi için görüntü üzerinde 2 köşe noktasını seçmeniz gerekecek.")
            print("OpenCV penceresi açılacak ve fare ile nokta seçimi yapabileceksiniz.")
            print("İşlemi tamamlamak için ENTER tuşuna, iptal etmek için ESC tuşuna basın.")

            secilen_noktalar = []

            def nokta_sec(event, x, y, flags, param):
                nonlocal secilen_noktalar, cizim_goruntu

                if event == cv2.EVENT_LBUTTONDOWN:
                    if len(secilen_noktalar) < 2:
                        secilen_noktalar.append((x, y))
                        print(f"Nokta {len(secilen_noktalar)} Seçildi: {x}, {y}")

                        cizim_goruntu = bgr_img.copy()
                        for i, (px, py) in enumerate(secilen_noktalar):
                            cv2.circle(cizim_goruntu, (px, py), 5, (0, 0, 255), -1)
                            cv2.putText(cizim_goruntu, f"{i + 1}", (px + 10, py + 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                        if len(secilen_noktalar) == 2:

                            try:
                                renk = self.cizim_rengi.get()
                                if renk == "red":
                                    color = (0, 0, 255)
                                elif renk == "green":
                                    color = (0, 255, 0)
                                elif renk == "blue":
                                    color = (255, 0, 0)
                                else:
                                    color = (0, 0, 255)
                            except:
                                color = (0, 0, 255)


                            try:
                                kalinlik = self.kalinlik_scale_value.get()
                            except:
                                kalinlik = 2

                            pt1 = secilen_noktalar[0]
                            pt2 = secilen_noktalar[1]
                            cv2.rectangle(cizim_goruntu, pt1, pt2, color, kalinlik)

                        cv2.imshow(pencere_adi, cizim_goruntu)

            cv2.namedWindow(pencere_adi)
            cv2.setMouseCallback(pencere_adi, nokta_sec)
            cv2.imshow(pencere_adi, bgr_img)

            while True:
                key = cv2.waitKey(0) & 0xFF

                # ESC ile iptal
                if key == 27:  # ESC
                    print("Dikdörtgen çizme işlemi iptal edildi.")
                    cv2.destroyWindow(pencere_adi)
                    return

                # ENTER ile tamamla
                elif key == 13:  # ENTER
                    if len(secilen_noktalar) == 2:
                        self.working_image = cv2.cvtColor(cizim_goruntu, cv2.COLOR_BGR2RGB)
                        self.goruntu_guncelle()
                        print("Dikdörtgen çizme işlemi tamamlandı.")
                        break
                    else:
                        print(f"Lütfen 2 nokta seçin! Şu ana kadar {len(secilen_noktalar)} nokta seçildi.")

            cv2.destroyWindow(pencere_adi)

        except Exception as e:
            print(f"Hata: Dikdörtgen çizme işlemi sırasında bir hata oluştu: {str(e)}")
            traceback.print_exc()
            cv2.destroyAllWindows()

    def daire_ciz_uygula(self):
        if self.working_image is None:
            print("Daire Çiz - Hata: İşlem yapılacak bir görüntü yok!")
            messagebox.showerror("Hata", "İşlem yapılacak bir görüntü yok! Lütfen önce bir görüntü yükleyin.")
            return

        try:
            bgr_img = cv2.cvtColor(self.working_image, cv2.COLOR_RGB2BGR)
            cizim_goruntu = bgr_img.copy()
            pencere_adi = "Daire Çizme - İki nokta seçin (İptal: ESC, Tamamla: ENTER)"
            print("Not: Daire çizme işlemi için görüntü üzerinde 2 nokta seçmeniz gerekecek.")
            print("İlk nokta dairenin merkezi, ikinci nokta yarıçapını belirleyecektir.")
            print("OpenCV penceresi açılacak ve fare ile nokta seçimi yapabileceksiniz.")
            print("İşlemi tamamlamak için ENTER tuşuna, iptal etmek için ESC tuşuna basın.")
            secilen_noktalar = []

            def nokta_sec(event, x, y, flags, param):
                nonlocal secilen_noktalar, cizim_goruntu

                if event == cv2.EVENT_LBUTTONDOWN:

                    if len(secilen_noktalar) < 2:
                        secilen_noktalar.append((x, y))
                        print(f"Nokta {len(secilen_noktalar)} Seçildi: {x}, {y}")


                        cizim_goruntu = bgr_img.copy()
                        for i, (px, py) in enumerate(secilen_noktalar):
                            cv2.circle(cizim_goruntu, (px, py), 5, (0, 0, 255), -1)
                            cv2.putText(cizim_goruntu, f"{i + 1}", (px + 10, py + 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                        if len(secilen_noktalar) == 2:

                            try:
                                renk = self.cizim_rengi.get()
                                if renk == "red":
                                    color = (0, 0, 255)
                                elif renk == "green":
                                    color = (0, 255, 0)
                                elif renk == "blue":
                                    color = (255, 0, 0)
                                else:
                                    color = (0, 0, 255)
                            except:
                                color = (0, 0, 255)


                            try:
                                kalinlik = self.kalinlik_scale_value.get()
                            except:
                                kalinlik = 2

                            center = secilen_noktalar[0]
                            radius_point = secilen_noktalar[1]
                            radius = int(
                                ((center[0] - radius_point[0]) ** 2 + (center[1] - radius_point[1]) ** 2) ** 0.5)

                            cv2.circle(cizim_goruntu, center, radius, color, kalinlik)

                        cv2.imshow(pencere_adi, cizim_goruntu)

            cv2.namedWindow(pencere_adi)
            cv2.setMouseCallback(pencere_adi, nokta_sec)
            cv2.imshow(pencere_adi, bgr_img)

            while True:
                key = cv2.waitKey(0) & 0xFF

                # ESC ile iptal
                if key == 27:  # ESC
                    print("Daire çizme işlemi iptal edildi.")
                    cv2.destroyWindow(pencere_adi)
                    return

                # ENTER ile tamamla
                elif key == 13:  # ENTER
                    if len(secilen_noktalar) == 2:
                        self.working_image = cv2.cvtColor(cizim_goruntu, cv2.COLOR_BGR2RGB)
                        self.goruntu_guncelle()
                        print("Daire çizme işlemi tamamlandı.")
                        break
                    else:
                        print(f"Lütfen 2 nokta seçin! Şu ana kadar {len(secilen_noktalar)} nokta seçildi.")

            cv2.destroyWindow(pencere_adi)

        except Exception as e:
            print(f"Hata: Daire çizme işlemi sırasında bir hata oluştu: {str(e)}")
            traceback.print_exc()
            cv2.destroyAllWindows()

def gabor_filtre(img, kernel_size=21, sigma=5, theta=np.pi / 4, lambd=10, gamma=0.5, psi=0, normalize=True):

    if img is None:
        print("Hata: İşlenecek görüntü bulunamadı!")
        return None

    try:

        if len(img.shape) > 2:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()

        kernel = cv2.getGaborKernel(
            (kernel_size, kernel_size),
            sigma,
            theta,
            lambd,
            gamma,
            psi,
            ktype=cv2.CV_32F
        )

        kernel /= np.sum(np.abs(kernel))
        filtered_img = cv2.filter2D(gray, cv2.CV_8UC3, kernel)
        if normalize:
            filtered_img = cv2.normalize(filtered_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        return filtered_img

    except Exception as e:
        print(f"Hata: Gabor filtresi uygulanırken bir hata oluştu: {str(e)}")
        return None

if __name__ == "__main__":
    root = tk.Tk()
    app = GoruntuIslemeUygulamasi(root)
    root.mainloop()