import torch
import sys

def check_gpu_availability():
    """
    Kiểm tra xem PyTorch có thể truy cập GPU (CUDA) hay không
    và in ra thông tin chi tiết.
    """
    print("==========================================")
    print("   BẮT ĐẦU KIỂM TRA GPU (PyTorch)   ")
    print("==========================================")
    
    try:
        # Kiểm tra phiên bản PyTorch
        print(f"Phiên bản PyTorch:      {torch.__version__}")
        
        # 1. Kiểm tra quan trọng nhất: CUDA có sẵn không?
        is_available = torch.cuda.is_available()
        print(f"CUDA (GPU) có sẵn?:    {is_available}")

        if not is_available:
            print("\n[CẢNH BÁO] ❌: PyTorch không tìm thấy GPU.")
            print("Vui lòng kiểm tra các lý do sau:")
            print("  1. Bạn chưa cài đặt driver NVIDIA mới nhất.")
            print("  2. Bạn đã cài nhầm phiên bản PyTorch (bản CPU).")
            print("     -> Gỡ ra và cài lại bản GPU từ trang chủ PyTorch.")
            print("  3. (Nếu có) Môi trường venv của bạn chưa được kích hoạt.")
            print("==========================================")
            return False

        # 2. Lấy thông tin chi tiết nếu CUDA có sẵn
        print("--- Thông tin GPU ---")
        
        # Lấy số lượng GPU
        gpu_count = torch.cuda.device_count()
        print(f"Số lượng GPU tìm thấy: {gpu_count}")

        # Lấy tên của GPU 0 (GPU mặc định)
        gpu_name = torch.cuda.get_device_name(0)
        print(f"Tên GPU (Thiết bị 0): {gpu_name}")
        
        # Lấy phiên bản CUDA mà PyTorch được build
        torch_cuda_version = torch.version.cuda
        print(f"Phiên bản CUDA (torch): {torch_cuda_version}")

        print("\n[THÀNH CÔNG] ✅: Mọi thứ đã sẵn sàng để train bằng GPU!")
        print("==========================================")
        return True

    except ImportError:
        print("\n[LỖI] ❌: Không tìm thấy thư viện 'torch' (PyTorch).")
        print("Vui lòng kích hoạt venv và chạy 'pip install torch' trước.")
        print("==========================================")
        return False
    except Exception as e:
        print(f"\n[LỖI] ❌: Đã xảy ra lỗi không xác định: {e}")
        print("==========================================")
        return False

if __name__ == "__main__":
    check_gpu_availability()