# ==============================================================================
# FILE 1: train.py
# TUJUAN: Mengunduh dataset dan melatih model YOLOv11.
# CARA PAKAI:
# 1. Ganti tulisan "GANTI_DENGAN_API_KEY_RAHASIA_ANDA" dengan Private API Key Anda.
# 2. Jalankan file ini dari terminal: python train.py
# ==============================================================================

from ultralytics import YOLO
from roboflow import Roboflow
import os

# --- BAGIAN KONFIGURASI (Hanya ubah API Key) ---

# Ganti dengan Private API Key BARU milik Anda.
ROBOFLOW_API_KEY = "6hDs2CDtacN8OhSoicXg"

# Informasi Dataset dari Roboflow (berdasarkan screenshot Anda)
WORKSPACE_ID = "Yen-an-Universe"
PROJECT_ID = "pattn-only"
VERSION_NUMBER = 1

# Pengaturan Training
MODEL_ARCHITECTURE = 'yolo11n.pt'  # 'n' untuk nano, paling cepat.
EPOCHS = 75                      # Jumlah siklus training.
IMAGE_SIZE = 640                 # Ukuran gambar yang dipakai.
PROJECT_NAME = 'yolo11n_jari_custom'  # Nama folder hasil training.

# --- BAGIAN EKSEKUSI (Jangan diubah) ---

# Inisialisasi Roboflow
print("Menghubungi Roboflow untuk mengunduh dataset...")
try:
    rf = Roboflow(api_key=ROBOFLOW_API_KEY)
    project = rf.workspace(WORKSPACE_ID).project(PROJECT_ID)
    dataset = project.version(VERSION_NUMBER).download("yolov8")
    print(f"Dataset berhasil diunduh ke lokasi: {dataset.location}")
except Exception as e:
    print(f"GAGAL: Terjadi error saat mengunduh dataset.")
    print(f"Pastikan API Key Anda sudah benar dan memiliki izin.")
    print(f"Detail Error: {e}")
    exit()

# Muat arsitektur model dasar
model = YOLO(MODEL_ARCHITECTURE)

# Mulai proses training
print("\nMemulai proses training model. Ini akan memakan waktu...")
results = model.train(
    data=os.path.join(dataset.location, 'data.yaml'),
    epochs=EPOCHS,
    imgsz=IMAGE_SIZE,
    name=PROJECT_NAME
)

# Pesan setelah training selesai
print("\n" + "="*50)
print("TRAINING SELESAI!")
model_path = f"runs/detect/{PROJECT_NAME}/weights/best.pt"
print(f"Model terbaik Anda berhasil dibuat dan disimpan di:")
print(model_path)
print("\nLangkah selanjutnya:")
print("1. Salin path model di atas.")
print("2. Buka file 'run_webcam.py'.")
print("3. Tempel path tersebut ke variabel 'MODEL_PATH'.")
print("4. Jalankan 'run_webcam.py' untuk melihat hasilnya di webcam!")
print("="*50)
