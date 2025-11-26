import os
from ultralytics import YOLO

def main():
    print("Bắt đầu quá trình huấn luyện...")

    # --- CÁC THAM SỐ HUẤN LUYỆN ---
    
    # 1. Chỉ định mô hình gốc
    MODEL_NAME = 'yolov8s.pt'

    # 2. Chỉ định file data.yaml
    DATA_CONFIG_PATH = 'data.yaml'

    # 3. Các tham số train chính
    EPOCHS = 100               # Số lượng epochs (lần học)
    IMG_SIZE = 640             # Kích thước ảnh đầu vào
    BATCH_SIZE = 16            # Số lượng ảnh xử lý trong 1 lần (điều chỉnh nếu bị lỗi "Out of Memory")
    PROJECT_NAME = 'runs'      # Tên thư mục gốc chứa kết quả
    RUN_NAME = 'run_1_yolov8s' # Tên thư mục con cho lần chạy này

    # --- KẾT THÚC THAM SỐ ---

    # Tải mô hình
    # Nếu bạn muốn tiếp tục train từ một checkpoint, thay MODEL_NAME bằng đường dẫn tới file .pt
    # ví dụ: 'runs/detect/train/weights/last.pt'
    model = YOLO(MODEL_NAME)

    # Bắt đầu huấn luyện
    print(f"Đang huấn luyện mô hình {MODEL_NAME} với file data {DATA_CONFIG_PATH}...")
    
    model.train(
        data=DATA_CONFIG_PATH,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        project=PROJECT_NAME,
        name=RUN_NAME,
        patience=30,
        device=0,
        workers=0
    )

    print("--- HUẤN LUYỆN HOÀN TẤT ---")
    print(f"Kết quả được lưu tại thư mục: {PROJECT_NAME}/{RUN_NAME}")

if __name__ == '__main__':
    main()