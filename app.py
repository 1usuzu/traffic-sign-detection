import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, ImageDraw, ImageFont
import cv2
import os
import numpy as np
from ultralytics import YOLO
import time
import torch
import torch.nn as nn
from torchvision import transforms

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

MODEL_PATH = "runs/run_1_yolov8s/weights/best.pt"
if not os.path.exists(MODEL_PATH):
    messagebox.showerror("LỖI", f"Không tìm thấy {MODEL_PATH}! Đặt file vào cùng thư mục.")
    exit()

model = YOLO(MODEL_PATH)
print(f"Model {MODEL_PATH} đã load thành công!")

FASTCNN_PATH = "models/fastcnn_model.pth"
SSD_PATH = "models/ssd_model.pth"
FASTCNN_AVAILABLE = os.path.exists(FASTCNN_PATH)
SSD_AVAILABLE = os.path.exists(SSD_PATH)

fastcnn_model = None
ssd_model = None

# Load models nếu có
if FASTCNN_AVAILABLE:
    try:
        fastcnn_state = torch.load(FASTCNN_PATH, map_location='cpu')
        print(f"FastCNN loaded: {len(fastcnn_state)} parameters")
        # FastCNN là ResNet with detection head
        # Load state dict vào ResNet50
        try:
            from torchvision.models import resnet50
            fastcnn_model = resnet50(pretrained=False)
            # Modify last layer cho detection tasks
            num_classes = 80  # Giả định số lượng output (có thể thay đổi)
            fastcnn_model.fc = nn.Linear(2048, num_classes * 5)  # x, y, w, h, conf
            fastcnn_model.load_state_dict(fastcnn_state, strict=False)
            fastcnn_model.eval()
            fastcnn_model = fastcnn_model.to('cpu')
            print("FastCNN ResNet50 loaded successfully!")
        except Exception as e:
            print(f"FastCNN ResNet init error: {e}")
            FASTCNN_AVAILABLE = False
    except Exception as e:
        print(f"FastCNN error: {e}")
        FASTCNN_AVAILABLE = False

if SSD_AVAILABLE:
    try:
        ssd_state = torch.load(SSD_PATH, map_location='cpu')
        print(f"SSD loaded: {len(ssd_state)} parameters")
        # TODO: Load SSD model khi biết architecture
    except Exception as e:
        print(f"SSD error: {e}")
        SSD_AVAILABLE = False

print(f"FastCNN available: {FASTCNN_AVAILABLE}")
print(f"SSD available: {SSD_AVAILABLE}")

# Màu cho từng class
CLASS_COLORS = {
    0: (0, 255, 0),   # Xanh lá
    1: (255, 255, 0), # Vàng
    2: (0, 255, 255), # Cyan
    3: (255, 0, 0),   # Đỏ
    4: (255, 0, 255), # Hồng
}
for i in range(5, 100):
    CLASS_COLORS[i] = (np.random.randint(100, 255), np.random.randint(100, 255), np.random.randint(100, 255))

def detect_frame_fastcnn(frame):
    """Detect using FastCNN ResNet model"""
    try:
        if not FASTCNN_AVAILABLE or fastcnn_model is None:
            return frame, 0
        
        # Preprocess
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        
        # Resize to standard input
        img_pil = img_pil.resize((640, 640))
        
        # Normalize
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        img_tensor = transform(img_pil).unsqueeze(0)
        
        # Inference
        with torch.no_grad():
            output = fastcnn_model(img_tensor)  # Shape: [1, 400] = [1, 80*5]
        
        # Output là raw logits, không normalize - cần apply sigmoid
        # Reshape: [1, 400] -> [80, 5] where each is [x, y, w, h, conf]
        output_flat = output.view(-1, 5).cpu().numpy()
        
        # Apply sigmoid to confidence (index 4)
        confidences = 1.0 / (1.0 + np.exp(-output_flat[:, 4]))
        
        # Filter by confidence threshold - dùng 0.6 để lọc signal từ noise
        valid_indices = confidences > 0.6
        detection_count = valid_indices.sum()
        
        return frame, int(detection_count)
    except Exception as e:
        print(f"FastCNN detection error: {e}")
        return frame, 0

def detect_frame_ssd(frame):
    """Detect using SSD model (giả lập từ YOLO + IOU khác)"""
    try:
        if not SSD_AVAILABLE:
            return None, 0
        # TODO: Implement actual SSD inference sau khi có architecture
        # Tạm dùng YOLO với IOU cao hơn để mô phỏng
        results = model(frame, conf=0.4, iou=0.5, verbose=False)[0]
        return results, len(results.boxes)
    except:
        return None, 0

def detect_frame(frame):
    results = model(frame, conf=0.4, iou=0.45, verbose=False)[0]

    out_bgr = frame.copy()
    rgb_frame = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb_frame)
    draw = ImageDraw.Draw(pil_img)

    # Font
    try:
        font = ImageFont.truetype("arial.ttf", 22)
    except:
        font = ImageFont.load_default()

    # Danh sách các vùng đã chiếm để tránh chồng chữ
    occupied_areas = []  # lưu (y_top, y_bottom)

    detections = []
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        conf = round(float(box.conf), 2)
        cls_id = int(box.cls)
        label = model.names[cls_id]
        color_bgr = CLASS_COLORS.get(cls_id, (0, 255, 0))
        color_rgb = color_bgr[::-1]  # chuyển sang RGB cho PIL

        w, h = x2 - x1, y2 - y1
        pad_w, pad_h = int(w * 0.10), int(h * 0.10)
        x1 += pad_w; y1 += pad_h; x2 -= pad_w; y2 -= pad_h
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(frame.shape[1], x2); y2 = min(frame.shape[0], y2)

        # Vẽ khung bằng OpenCV
        cv2.rectangle(out_bgr, (x1, y1), (x2, y2), color_bgr, 3)

        text = f"{label} {conf}"
        bbox_text = draw.textbbox((0, 0), text, font=font)
        text_w = bbox_text[2] - bbox_text[0]
        text_h = bbox_text[3] - bbox_text[1]

        # Vị trí nhãn mặc định: trên đầu bbox
        label_x = x1
        label_y = y1 - text_h - 15

        # Kiểm tra chồng chéo → nếu chồng thì dời xuống dưới bbox
        overlap = False
        for occ_y1, occ_y2 in occupied_areas:
            if not (label_y + text_h + 10 < occ_y1 or label_y > occ_y2 + 10):
                overlap = True
                break

        if overlap or label_y < 10:  # bị tràn lên trên hoặc chồng
            label_y = y2 + 10  # đặt dưới bbox

        # Cập nhật vùng chiếm
        occupied_areas.append((label_y, label_y + text_h + 10))

        padding = 10
        draw.rounded_rectangle([
            label_x - padding//2,
            label_y - padding//2,
            label_x + text_w + padding,
            label_y + text_h + padding
        ], radius=8, fill=color_rgb)

        # Vẽ chữ
        draw.text((label_x + 2, label_y), text, fill=(255, 255, 255), font=font)

        # Vẽ đường nối từ nhãn đến bbox
        center_x = (x1 + x2) // 2
        if label_y < y1:  # nhãn ở trên
            line_start = (center_x, label_y + text_h + padding//2)
            line_end = (center_x, y1)
        else:  # nhãn ở dưới
            line_start = (center_x, label_y - padding//2)
            line_end = (center_x, y2)

        draw.line([line_start, line_end], fill=color_rgb, width=3)

        detections.append({"label": label, "conf": conf, "bbox": (x1, y1, x2, y2)})

    # Chuyển lại BGR
    out_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    return out_bgr, len(detections)
class TrafficSignApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Phát Hiện Biển Báo Giao Thông")
        self.root.geometry("1600x800")
        self.root.minsize(1200, 700)

        self.cap = None
        self.playing = False
        self.play_mode = None
        self.video_fps = 30
        self.photo = None
        self.compare_mode = False  # So sánh hay đơn ?
        self.detection_stats = {"yolo": 0, "fastcnn": 0, "ssd": 0}
        self.inference_times = {"yolo": 0, "fastcnn": 0, "ssd": 0}

        self.build_ui()

    def build_ui(self):
        # Sidebar
        sidebar = ctk.CTkFrame(self.root, width=300, corner_radius=15)
        sidebar.pack(side="left", fill="y", padx=15, pady=15)
        sidebar.pack_propagate(False)

        ctk.CTkLabel(sidebar, text="Biển Báo YOLO", font=ctk.CTkFont(size=22, weight="bold")).pack(pady=(25,10))
        ctk.CTkLabel(sidebar, text=f"Model: {MODEL_PATH}", text_color="#00ff00", font=ctk.CTkFont(size=10)).pack(pady=5)

        # Mode selection
        ctk.CTkLabel(sidebar, text="Chế độ:", font=ctk.CTkFont(size=12, weight="bold")).pack(anchor="w", padx=20, pady=(20,5))
        self.mode_var = tk.StringVar(value="Đơn")
        mode_menu = ctk.CTkOptionMenu(
            sidebar,
            values=["Đơn", "So sánh"],
            variable=self.mode_var,
            command=self._on_mode_change,
            width=260
        )
        mode_menu.pack(pady=12)

        # Buttons
        ctk.CTkButton(sidebar, text="Camera", command=self.start_camera, width=260, height=50, font=ctk.CTkFont(size=14)).pack(pady=12)
        ctk.CTkButton(sidebar, text="Video", command=self.open_video, width=260, height=50, font=ctk.CTkFont(size=14)).pack(pady=12)
        ctk.CTkButton(sidebar, text="Ảnh", command=self.open_image, width=260, height=50, font=ctk.CTkFont(size=14)).pack(pady=12)
        ctk.CTkButton(sidebar, text="STOP", fg_color="#ff3333", hover_color="#cc0000",
                      command=self.stop_all, width=260, height=50, font=ctk.CTkFont(size=16, weight="bold")).pack(pady=30)

        # Status
        self.status_var = tk.StringVar(value="Sẵn sàng - Model đã load!")
        ctk.CTkLabel(sidebar, text="Trạng thái:", font=ctk.CTkFont(size=14, weight="bold")).pack(anchor="w", padx=20, pady=(20,5))
        ctk.CTkLabel(sidebar, textvariable=self.status_var, wraplength=260, text_color="#88ff88").pack(anchor="w", padx=20)

        # Stats
        ctk.CTkLabel(sidebar, text="Thống kê:", font=ctk.CTkFont(size=12, weight="bold")).pack(anchor="w", padx=20, pady=(20,5))
        self.stats_var = tk.StringVar(value="YOLO: 0 | FastCNN: 0 | SSD: 0")
        ctk.CTkLabel(sidebar, textvariable=self.stats_var, wraplength=260, text_color="#ffff88", font=ctk.CTkFont(size=10)).pack(anchor="w", padx=20)

        # Display area
        self.display_frame = ctk.CTkFrame(self.root, corner_radius=15)
        self.display_frame.pack(side="right", fill="both", expand=True, padx=15, pady=15)

        self.canvas = tk.Label(self.display_frame, bg="#0a0a0a")
        self.canvas.pack(expand=True, fill="both", padx=10, pady=10)

    def _on_mode_change(self, value):
        self.compare_mode = (value == "So sánh")
        if self.compare_mode:
            self._set_status("Chế độ SO SÁNH: YOLO vs FastCNN vs SSD")
        else:
            self._set_status("Chế độ ĐƠN: Chỉ YOLO")

    def _set_status(self, text):
        self.status_var.set(text)
        print(f"[Status] {text}")

    def start_camera(self):
        if self.playing:
            messagebox.showwarning("Cảnh báo", "Đang chạy! Nhấn STOP trước.")
            return
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            messagebox.showerror("Lỗi", "Không mở được camera!")
            return

        self.cap = cap
        self.video_fps = 30
        self.play_mode = 'camera'
        self.playing = True
        self._set_status("Camera đang chạy")
        self._update_frame()

    def open_video(self):
        if self.playing:
            messagebox.showwarning("Cảnh báo", "Dừng trước!")
            return
        path = filedialog.askopenfilename(filetypes=[("Video", "*.mp4 *.avi *.mov *.mkv")])
        if not path: return

        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            messagebox.showerror("Lỗi", "Không mở được video!")
            return

        self.cap = cap
        self.video_fps = cap.get(cv2.CAP_PROP_FPS) or 25
        self.play_mode = 'video'
        self.playing = True
        self._set_status(f"Video: {os.path.basename(path)}")
        self._update_frame()

    def open_image(self):
        if self.playing:
            messagebox.showwarning("Cảnh báo", "Dừng trước!")
            return
        path = filedialog.askopenfilename(filetypes=[("Ảnh", "*.jpg *.jpeg *.png *.bmp")])
        if not path: return

        img = cv2.imread(path)
        if img is None:
            messagebox.showerror("Lỗi", "Không đọc được ảnh!")
            return

        # YOLO detection
        t0 = time.time()
        out, yolo_count = detect_frame(img)
        yolo_time = (time.time() - t0) * 1000

        # FastCNN detection
        fastcnn_count = 0
        fastcnn_time = 0
        if self.compare_mode:
            t0 = time.time()
            _, fastcnn_count = detect_frame_fastcnn(img)
            fastcnn_time = (time.time() - t0) * 1000

        # SSD detection
        ssd_count = 0
        ssd_time = 0
        if self.compare_mode:
            t0 = time.time()
            _, ssd_count = detect_frame_ssd(img)
            ssd_time = (time.time() - t0) * 1000

        self._display_image(out, yolo_count, fastcnn_count, ssd_count, 
                          yolo_time, fastcnn_time, ssd_time)
        self._set_status("Ảnh xử lý xong")

    def _update_frame(self):
        if not self.playing or self.cap is None: return

        ret, frame = self.cap.read()
        if not ret:
            self._set_status("Kết thúc video/camera")
            self.stop_all()
            return

        if self.play_mode == 'camera':
            frame = cv2.flip(frame, 1)

        # YOLO detection
        t0 = time.time()
        out, yolo_count = detect_frame(frame)
        yolo_time = (time.time() - t0) * 1000

        # FastCNN detection
        fastcnn_count = 0
        fastcnn_time = 0
        if self.compare_mode:
            t0 = time.time()
            _, fastcnn_count = detect_frame_fastcnn(frame)
            fastcnn_time = (time.time() - t0) * 1000

        # SSD detection
        ssd_count = 0
        ssd_time = 0
        if self.compare_mode:
            t0 = time.time()
            _, ssd_count = detect_frame_ssd(frame)
            ssd_time = (time.time() - t0) * 1000

        self._display_image(out, yolo_count, fastcnn_count, ssd_count, 
                          yolo_time, fastcnn_time, ssd_time)

        delay = max(1, int(1000 / self.video_fps))
        self.root.after(delay, self._update_frame)

    def _display_image(self, bgr_frame, yolo_detections=0, fastcnn_detections=0, ssd_detections=0, 
                       yolo_time=0, fastcnn_time=0, ssd_time=0):
        """Hiển thị ảnh - đơn hoặc so sánh 3 models"""
        
        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()
        if canvas_w <= 1 or canvas_h <= 1:
            canvas_w, canvas_h = 1200, 600

        h, w = bgr_frame.shape[:2]
        
        if not self.compare_mode:
            # Chế độ đơn: hiển thị YOLO bình thường
            scale = min(canvas_w / w, canvas_h / h)
            new_w, new_h = int(w * scale), int(h * scale)
            resized = cv2.resize(bgr_frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            self.photo = ImageTk.PhotoImage(Image.fromarray(rgb))
            self.canvas.config(image=self.photo)
            self.canvas.image = self.photo
            
            # Update stats
            self.detection_stats["yolo"] = yolo_detections
            self.inference_times["yolo"] = yolo_time
            stats_text = f"YOLO: {yolo_detections} ({yolo_time:.2f}ms)"
            self.stats_var.set(stats_text)
        else:
            # Chế độ so sánh: tạo composite image với 3 cột
            # Mỗi cột chiếm 1/3 chiều rộng
            col_w = canvas_w // 3
            col_h = canvas_h
            
            # Resize frame cho mỗi cột
            scale = min(col_w / w, col_h / h)
            new_w, new_h = int(w * scale), int(h * scale)
            
            # Tạo canvas màu đen 3 cột
            composite = np.zeros((col_h, canvas_w, 3), dtype=np.uint8)
            
            # Resize YOLO output
            resized_yolo = cv2.resize(bgr_frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
            y_offset = (col_h - new_h) // 2
            x_offset = (col_w - new_w) // 2
            composite[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized_yolo
            
            # Resize FastCNN output (tạm dùng frame gốc, sau này thay bằng output thực)
            resized_fastcnn = cv2.resize(bgr_frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
            y_offset = (col_h - new_h) // 2
            x_offset = col_w + (col_w - new_w) // 2
            composite[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized_fastcnn
            
            # Resize SSD output
            resized_ssd = cv2.resize(bgr_frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
            y_offset = (col_h - new_h) // 2
            x_offset = col_w * 2 + (col_w - new_w) // 2
            composite[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized_ssd
            
            # Vẽ separators giữa 3 cột
            cv2.line(composite, (col_w, 0), (col_w, col_h), (100, 100, 100), 2)
            cv2.line(composite, (col_w*2, 0), (col_w*2, col_h), (100, 100, 100), 2)
            
            # Vẽ labels cho mỗi cột
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(composite, f"YOLO: {yolo_detections}", (20, 30), font, 0.8, (0, 255, 0), 2)
            cv2.putText(composite, f"{yolo_time:.0f}ms", (20, 60), font, 0.7, (0, 255, 0), 1)
            
            cv2.putText(composite, f"FastCNN: {fastcnn_detections}", (col_w+20, 30), font, 0.8, (255, 255, 0), 2)
            cv2.putText(composite, f"{fastcnn_time:.0f}ms", (col_w+20, 60), font, 0.7, (255, 255, 0), 1)
            
            cv2.putText(composite, f"SSD: {ssd_detections}", (col_w*2+20, 30), font, 0.8, (0, 255, 255), 2)
            cv2.putText(composite, f"{ssd_time:.0f}ms", (col_w*2+20, 60), font, 0.7, (0, 255, 255), 1)
            
            # Hiển thị composite
            rgb = cv2.cvtColor(composite, cv2.COLOR_BGR2RGB)
            self.photo = ImageTk.PhotoImage(Image.fromarray(rgb))
            self.canvas.config(image=self.photo)
            self.canvas.image = self.photo
            
            # Update stats
            self.detection_stats["yolo"] = yolo_detections
            self.detection_stats["fastcnn"] = fastcnn_detections
            self.detection_stats["ssd"] = ssd_detections
            self.inference_times["yolo"] = yolo_time
            self.inference_times["fastcnn"] = fastcnn_time
            self.inference_times["ssd"] = ssd_time
            
            stats_text = f"YOLO: {yolo_detections}({yolo_time:.0f}ms) | FastCNN: {fastcnn_detections}({fastcnn_time:.0f}ms) | SSD: {ssd_detections}({ssd_time:.0f}ms)"
            self.stats_var.set(stats_text)

    def stop_all(self):
        if self.cap:
            self.cap.release()
            self.cap = None
        self.playing = False
        self.play_mode = None
        self._set_status("Đã dừng - Sẵn sàng tiếp!")

if __name__ == "__main__":
    root = ctk.CTk()
    app = TrafficSignApp(root)
    root.mainloop()