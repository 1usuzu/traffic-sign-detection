"""
Script kiểm tra tính toàn vẹn của bộ dữ liệu YOLO

Cách dùng:
1. Đặt file này vào thư mục gốc của dataset (nơi chứa file data.yaml).
2. Chạy script từ terminal:
   python check_data.py
"""

import yaml
import os
import argparse
from pathlib import Path
from collections import Counter
from datetime import datetime

def find_label_path(img_path: Path) -> Path:
    """
    Suy ra đường dẫn thư mục 'labels' từ đường dẫn thư mục 'images'.

    """
    return img_path.parent / 'labels'

def check_split(img_dir_str: str, nc: int, dataset_root: Path) -> tuple:
    """
    Kiểm tra một tập dữ liệu con

    """
    img_path = (dataset_root / img_dir_str).resolve()
    lbl_path = find_label_path(img_path)
    
    report_lines = []
    
    if not lbl_path.exists():
        report_lines.append(f"    LỖI: Không tìm thấy thư mục nhãn: {lbl_path}\n")
        return 0, 0, False, Counter(), "\n".join(report_lines)

    # Đếm file ảnh (hỗ trợ các định dạng phổ biến)
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
    image_files = [p for p in img_path.glob('*') if p.suffix.lower() in image_extensions]
    label_files = list(lbl_path.glob('*.txt'))

    image_count = len(image_files)
    label_file_count = len(label_files)
    
    all_labels_valid = True
    class_counts = Counter()
    total_bboxes = 0

    # Đọc và xác thực từng file nhãn
    for lbl_file in label_files:
        with open(lbl_file, 'r') as f:
            for line in f:
                try:
                    parts = line.split()
                    if not parts:
                        continue # Bỏ qua dòng trống
                        
                    class_id = int(float(parts[0]))
                    
                    if 0 <= class_id < nc:
                        class_counts.update([class_id])
                        total_bboxes += 1
                    else:
                        all_labels_valid = False
                        report_lines.append(f"    LỖI NHÃN (ngoài phạm vi nc={nc}): class_id {class_id} trong file {lbl_file.name}")
                
                except (ValueError, IndexError):
                    all_labels_valid = False
                    report_lines.append(f"    LỖI NHÃN (định dạng sai): file {lbl_file.name} - line: '{line.strip()}'")

    # Tạo báo cáo cho split này
    if all_labels_valid:
        report_lines.append("    Tất cả nhãn hợp lệ.")
    else:
        report_lines.append("    PHÁT HIỆN NHÃN KHÔNG HỢP LỆ!")

    report_lines.append(f"    Số ảnh: {image_count}, Số file nhãn: {label_file_count}")

    if image_count != label_file_count:
        report_lines.append(f"    CẢNH BÁO: Số ảnh ({image_count}) không khớp số file nhãn ({label_file_count}).")

    return image_count, label_file_count, all_labels_valid, class_counts, "\n".join(report_lines) + "\n"


def main_check(yaml_file: str, output_file: str):
    """
    Hàm chính thực hiện kiểm tra và ghi báo cáo.
    """
    start_time = datetime.now()
    report = []
    report.append("============================================================")
    report.append(f"KIỂM TRA DATASET - {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("============================================================\n")

    yaml_path = Path(yaml_file)
    if not yaml_path.exists():
        report.append(f"LỖI: Không tìm thấy file {yaml_file}")
        print("\n".join(report))
        return

    try:
        with open(yaml_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
    except Exception as e:
        report.append(f"LỖI: Không thể đọc file {yaml_file}. Lỗi: {e}")
        print("\n".join(report))
        return

    dataset_root = yaml_path.parent
    
    nc = data.get('nc')
    names = data.get('names')
    
    report.append(f"File {yaml_file}:")
    if nc is None or names is None:
        report.append("  LỖI: 'nc' hoặc 'names' không có trong file yaml.")
        print("\n".join(report))
        return

    report.append(f"   nc: {nc}")
    report.append(f"   Số tên lớp: {len(names)}\n")

    total_errors = 0
    if nc != len(names):
        report.append(f"  LỖI: Số 'nc' ({nc}) không khớp số lượng tên lớp ({len(names)}).\n")
        total_errors += 1
    
    total_class_counts = Counter()
    splits_to_check = [('train', 'TRAIN'), ('val', 'VALID'), ('test', 'TEST')]

    for split_key, split_name in splits_to_check:
        report.append(f"--- KIỂM TRA SPLIT: {split_name} ---")
        img_path_str = data.get(split_key)
        
        if not img_path_str:
            report.append(f"    BỎ QUA: Không tìm thấy key '{split_key}' trong yaml.\n")
            continue
        
        img_c, lbl_c, valid, counts, split_report_str = check_split(img_path_str, nc, dataset_root)
        
        report.append(split_report_str)
        if not valid:
            total_errors += 1
        total_class_counts.update(counts)

    report.append("============================================================")
    report.append("THỐNG KÊ SỐ LƯỢNG MỖI LỚP (TỔNG train + valid + test)")
    report.append("============================================================\n")

    missing_classes = []
    for i in range(nc):
        count = total_class_counts.get(i, 0)
        status = "OK" if count > 0 else "MISSING"
        name = names[i] if i < len(names) else "TÊN BỊ THIẾU"
        
        if count == 0:
            missing_classes.append(f"   - Lớp {i}: {name}")

        # Định dạng giống file mẫu: {index} | {count} | {status} | {name}
        report.append(f"{i:>2} | {count:>5} | {status:<7} | {name}")
    
    if missing_classes:
        total_errors += len(missing_classes) # Coi lớp bị thiếu là một lỗi

    report.append("\n============================================================")
    report.append("TỔNG KẾT")
    report.append("============================================================\n")

    if missing_classes:
        report.append(f"  CẢNH BÁO: Có {len(missing_classes)} lớp KHÔNG có dữ liệu:")
        report.extend(missing_classes)
        report.append("")

    if total_errors > 0:
        report.append(f" PHÁT HIỆN {total_errors} LỖI.")
        report.append(" CẦN SỬA TRƯỚC KHI TRAIN!")
        report.append("   Xem chi tiết ở trên.\n")
    else:
        report.append(" TẤT CẢ ĐỀU HỢP LỆ.")
        report.append("   Sẵn sàng để train!\n")
        
    end_time = datetime.now()
    report.append("============================================================")
    report.append(f"HOÀN TẤT KIỂM TRA: {end_time.strftime('%H:%M:%S')}")
    
    # Ghi báo cáo ra file
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("\n".join(report))
        
        print(f"Đã lưu báo cáo thành công vào: {output_file}")
        print("\nNội dung báo cáo:\n")
        print("\n".join(report))
        
    except IOError as e:
        print(f"LỖI: Không thể ghi file báo cáo. Lỗi: {e}")
        print("\nNội dung báo cáo:\n")
        print("\n".join(report))

if __name__ == "__main__":
    # Cài đặt thư viện bắt buộc (nếu chưa có): pip install PyYAML
    
    parser = argparse.ArgumentParser(
        description="Kiểm tra file YAML và các file nhãn của dataset YOLO.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    parser.add_argument(
        '--yaml', 
        type=str, 
        default='data.yaml', 
        help="Đường dẫn đến file data.yaml cần kiểm tra.\n(Mặc định: data.yaml)"
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='check_report.txt',
        help="Tên file để lưu báo cáo.\n(Mặc định: check_report.txt)"
    )
    
    args = parser.parse_args()
    
    print(f"Bắt đầu kiểm tra file: {args.yaml}...")
    main_check(args.yaml, args.output)