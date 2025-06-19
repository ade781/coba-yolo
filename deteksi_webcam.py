# Import library yang diperlukan.
# Pastikan Anda sudah menginstal 'ultralytics', 'opencv-python', dan 'yt-dlp' yang terbaru.
import cv2
from ultralytics import YOLO
import subprocess  # Import library untuk menjalankan perintah terminal

# --- PENGATURAN ---
TARGET_OBJECTS = ["car", "motorcycle", "bus", "truck"]
YOUTUBE_URL = "https://www.youtube.com/watch?v=FTixwJ8b9j4&ab_channel=Kompas.com"

# --- FUNGSI UNTUK MENDAPATKAN URL STREAM ASLI ---


def get_stream_url(youtube_url):
    """Menggunakan yt-dlp untuk mendapatkan URL stream video M3U8/HLS secara langsung."""
    print("Menggunakan yt-dlp untuk mendapatkan URL stream langsung...")
    try:
        # Jalankan perintah 'yt-dlp --get-url' dan tangkap outputnya
        result = subprocess.run(
            ['yt-dlp', '--get-url', youtube_url],
            capture_output=True,
            text=True,
            check=True
        )
        stream_url = result.stdout.strip()
        print("Berhasil mendapatkan URL stream.")
        return stream_url
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"Error saat menjalankan yt-dlp: {e}")
        print("Pastikan yt-dlp terinstal dan bisa diakses dari terminal Anda.")
        return None


# Pesan untuk pengguna bahwa program dimulai
print("Mencoba memulai deteksi kendaraan dari stream CCTV...")
print(f"Hanya akan mendeteksi: {', '.join(TARGET_OBJECTS)}")
print(f"Sumber video: {YOUTUBE_URL}")
print("Tekan 'q' pada jendela video untuk menghentikan program.")

# Muat model YOLOv11n
try:
    print("Memuat model YOLOv11n...")
    model = YOLO('yolo11n.pt')
    class_names = model.names
    print("Model berhasil dimuat.")
except Exception as e:
    print(f"Gagal memuat model YOLO. Error: {e}")
    exit()

# Dapatkan URL stream langsung sebelum membukanya dengan OpenCV
stream_url = get_stream_url(YOUTUBE_URL)
if not stream_url:
    exit()

# Buka akses ke stream video menggunakan URL yang sudah didapat
print("Mencoba terhubung ke stream video...")
cap = cv2.VideoCapture(stream_url)
if not cap.isOpened():
    print(f"Error: Tidak bisa membuka stream video meskipun URL sudah didapat dari yt-dlp.")
    exit()
print("Berhasil terhubung ke stream.")

try:
    while True:
        success, frame = cap.read()
        if not success:
            print("Stream video berakhir atau terputus.")
            break

        results = model(frame, stream=True, verbose=False)
        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                class_name = class_names[cls_id]

                if class_name in TARGET_OBJECTS:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f'{class_name} {conf:.2f}'
                    cv2.putText(frame, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Deteksi Kendaraan CCTV - YOLOv11", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except Exception as e:
    print(f"\n--- TERJADI ERROR SAAT RUNTIME ---")
    print(f"Error: {e}")

finally:
    cap.release()
    cv2.destroyAllWindows()
    print("\nProgram dihentikan. Semua jendela telah ditutup.")
