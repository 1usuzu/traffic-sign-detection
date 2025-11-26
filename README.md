# Hệ thống Nhận diện Biển báo Giao thông Việt Nam

## Vietnamese Traffic Sign Detection System

Dự án xây dựng hệ thống nhận diện biển báo giao thông tại Việt Nam sử dụng mô hình **YOLOv8**. Hệ thống bao gồm quy trình huấn luyện mô hình, công cụ kiểm tra dữ liệu, và ứng dụng giao diện người dùng (GUI) để nhận diện thời gian thực qua Camera, Video hoặc Ảnh.

---

## Tính năng chính

### Nhận diện đa lớp

- Hỗ trợ nhận diện **58 loại biển báo giao thông** phổ biến tại Việt Nam
- Bao gồm: Biển cấm, biển hiệu lệnh, biển cảnh báo, biển chỉ dẫn

### Mô hình mạnh mẽ

- Sử dụng kiến trúc **YOLOv8s** cho tốc độ và độ chính xác cao
- Hỗ trợ GPU acceleration với CUDA
- Tối ưu hóa cho real-time detection

### Ứng dụng GUI trực quan

- Giao diện hiện đại với **CustomTkinter**
- **3 chế độ phát hiện:**
  - **Camera**: Nhận diện trực tiếp qua Webcam
  - **Video**: Upload và phân tích video file (.mp4, .avi)
  - **Ảnh**: Nhận diện trên ảnh tĩnh
- Chế độ so sánh model (YOLO vs FastCNN vs SSD) - nếu có

### Công cụ tiện ích

- Script kiểm tra tính toàn vẹn của dữ liệu
- Script kiểm tra khả năng hoạt động của GPU
- Báo cáo chi tiết về dataset

---

## Dữ liệu (Dataset)

Bộ dữ liệu được cấu hình trong file `data.yaml` và bao gồm:

| Tập dữ liệu    | Số lượng ảnh   |
| -------------- | -------------- |
| **Train**      | 9,536 ảnh      |
| **Validation** | 784 ảnh        |
| **Test**       | 613 ảnh        |
| **Tổng**       | **10,933 ảnh** |


## Công nghệ sử dụng

- **Deep Learning Framework**: [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- **GUI Framework**: [CustomTkinter](https://github.com/TomSchimansky/CustomTkinter)
- **Computer Vision**: OpenCV
- **Deep Learning**: PyTorch
- **Data Processing**: NumPy, Pillow
- **Dataset Source**: Roboflow (Vietnam Traffic Sign Detection v5)

---

## Đóng góp

Dự án này được thực hiện cho mục đích **nghiên cứu và học tập**.

Nếu bạn muốn đóng góp:

1. Fork repository
2. Tạo branch mới (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Mở Pull Request

---

## Liên hệ

Nếu có câu hỏi hoặc góp ý, vui lòng tạo issue trên GitHub repository.
