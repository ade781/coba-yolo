# ==============================================================================
# FILE 2: run_webcam.py
# TUJUAN: Menggunakan model yang sudah dilatih untuk deteksi real-time via webcam.
# CARA PAKAI:
# 1. Setelah training selesai, salin path model 'best.pt' yang muncul di terminal.
# 2. Tempel path tersebut untuk menggantikan nilai variabel MODEL_PATH di bawah ini.
# 3. Jalankan file ini dari terminal: python run_webcam.py
# ==============================================================================

import cv2
from ultralytics import YOLO

# --- BAGIAN KONFIGURASI ---

# Ganti dengan path ke model 'best.pt' hasil dari training Anda.
# Contoh: 'runs/detect/yolo11n_jari_custom/weights/best.pt'
MODEL_PATH = 'runs/detect/yolo11n_jari_custom/weights/best.pt'

# --- BAGIAN EKSEKUSI (Jangan diubah) ---

# Muat model yang sudah dilatih
try:
    model = YOLO(MODEL_PATH)
except Exception as e:
    print(f"GAGAL: Tidak bisa memuat model dari path: {MODEL_PATH}")
    print(f"Pastikan path sudah benar dan file 'best.pt' ada di sana.")
    print(f"Detail Error: {e}")
    exit()

# Buka koneksi ke webcam (0 biasanya webcam internal)
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Tidak bisa membuka webcam.")
    exit()

print("Webcam terbuka. Arahkan tangan Anda ke kamera.")
print("Tekan tombol 'q' untuk keluar.")

# Loop untuk memproses setiap frame dari webcam
while True:
    success, frame = cap.read()
    if not success:
        print("Gagal membaca frame dari webcam. Keluar.")
        break

    # Lakukan deteksi objek dengan model Anda
    # stream=True direkomendasikan untuk video agar lebih efisien
    # verbose=False agar tidak menampilkan banyak log di terminal
    results = model(frame, stream=True, verbose=False)

    object_count = 0
    for r in results:
        boxes = r.boxes
        for box in boxes:
            object_count += 1  # Hitung setiap objek yang terdeteksi

            # Ambil koordinat kotak pembatas (bounding box)
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Gambar kotak di frame
            # Warna ungu, ketebalan 2
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)

            # Siapkan teks label (nama kelas dan tingkat kepercayaan)
            conf = box.conf[0]
            cls = int(box.cls[0])
            label = f'{model.names[cls]} {conf:.2f}'

            # Tulis teks label di atas kotak
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

    # Tampilkan jumlah total objek yang terdeteksi di pojok layar
    cv2.putText(frame, f'Jumlah Objek: {object_count}',
                (30, 50), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (0, 255, 0), 2)

    # Tampilkan jendela hasil
    cv2.imshow("YOLOv11 Deteksi Real-time", frame)

    # Cek jika tombol 'q' ditekan untuk keluar
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Tutup semua sumber daya
cap.release()
cv2.destroyAllWindows()
print("Aplikasi ditutup.")
