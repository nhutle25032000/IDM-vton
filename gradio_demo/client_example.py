#!/usr/bin/env python3
"""
Ví dụ về cách sử dụng IDM-VTON API từ Python client
"""
import requests
import base64
import io
from PIL import Image
import json

# Cấu hình API
API_BASE_URL = "http://localhost:8000"
API_ENDPOINT = f"{API_BASE_URL}/try-on"

def image_to_bytes(image_path):
    """Chuyển đổi file ảnh thành bytes"""
    with open(image_path, 'rb') as f:
        return f.read()

def base64_to_image(base64_str, save_path):
    """Chuyển đổi base64 string thành ảnh và lưu file"""
    image_data = base64.b64decode(base64_str)
    image = Image.open(io.BytesIO(image_data))
    image.save(save_path)
    print(f"✅ Đã lưu ảnh: {save_path}")

def test_api_connection():
    """Kiểm tra kết nối API"""
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ API đang hoạt động: {data}")
            return True
        else:
            print(f"❌ API không phản hồi: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Không thể kết nối API: {e}")
        return False

def try_on_api(human_image_path, garment_image_path, garment_description, 
               mask_image_path=None, use_auto_mask=True, use_auto_crop=False, 
               denoise_steps=30, seed=42):
    """
    Gọi API try-on
    
    Args:
        human_image_path: Đường dẫn ảnh người
        garment_image_path: Đường dẫn ảnh quần áo
        garment_description: Mô tả quần áo
        mask_image_path: Đường dẫn ảnh mask (tùy chọn)
        use_auto_mask: Sử dụng auto mask
        use_auto_crop: Sử dụng auto crop
        denoise_steps: Số bước denoise
        seed: Seed cho random generator
    """
    
    print(f"🚀 Bắt đầu try-on...")
    print(f"👤 Ảnh người: {human_image_path}")
    print(f"👕 Ảnh quần áo: {garment_image_path}")
    print(f"📝 Mô tả: {garment_description}")
    
    try:
        # Chuẩn bị files
        files = {
            'human_image': ('human.jpg', open(human_image_path, 'rb'), 'image/jpeg'),
            'garment_image': ('garment.jpg', open(garment_image_path, 'rb'), 'image/jpeg'),
        }
        
        # Thêm mask image nếu có
        if mask_image_path:
            files['mask_image'] = ('mask.jpg', open(mask_image_path, 'rb'), 'image/jpeg')
        
        # Chuẩn bị form data
        data = {
            'garment_description': garment_description,
            'use_auto_mask': use_auto_mask,
            'use_auto_crop': use_auto_crop,
            'denoise_steps': denoise_steps,
            'seed': seed
        }
        
        print("📤 Đang gửi request...")
        
        # Gọi API
        response = requests.post(API_ENDPOINT, files=files, data=data)
        
        # Đóng files
        for file_obj in files.values():
            if hasattr(file_obj[1], 'close'):
                file_obj[1].close()
        
        if response.status_code == 200:
            result = response.json()
            
            if result['success']:
                print("✅ Try-on thành công!")
                
                # Lưu ảnh kết quả
                base64_to_image(result['result_image'], 'result_output.png')
                base64_to_image(result['mask_image'], 'mask_output.png')
                
                return True
            else:
                print(f"❌ Try-on thất bại: {result.get('message', 'Unknown error')}")
                return False
        else:
            print(f"❌ API error: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Lỗi khi gọi API: {e}")
        return False

def main():
    """Hàm chính"""
    print("🌟 IDM-VTON API Client Demo")
    print("=" * 50)
    
    # Kiểm tra kết nối API
    if not test_api_connection():
        print("❌ Vui lòng chạy API server trước: python run_api.py")
        return
    
    # Đường dẫn ảnh mẫu (thay đổi theo ảnh của bạn)
    human_image_path = "example/human/00034_00.jpg"
    garment_image_path = "example/cloth/04469_00.jpg"
    garment_description = "white t-shirt"
    
    # Kiểm tra file tồn tại
    import os
    if not os.path.exists(human_image_path):
        print(f"❌ Không tìm thấy ảnh người: {human_image_path}")
        return
    
    if not os.path.exists(garment_image_path):
        print(f"❌ Không tìm thấy ảnh quần áo: {garment_image_path}")
        return
    
    # Gọi API
    success = try_on_api(
        human_image_path=human_image_path,
        garment_image_path=garment_image_path,
        garment_description=garment_description,
        use_auto_mask=True,
        use_auto_crop=False,
        denoise_steps=30,
        seed=42
    )
    
    if success:
        print("🎉 Demo hoàn thành! Kiểm tra file result_output.png và mask_output.png")
    else:
        print("❌ Demo thất bại!")

if __name__ == "__main__":
    main() 