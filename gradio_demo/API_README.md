# IDM-VTON API Documentation

API để sử dụng chức năng Virtual Try-on từ các ứng dụng bên ngoài.

## 🚀 Khởi chạy API Server

### 1. Cài đặt dependencies
```bash
pip install -r api_requirements.txt
```

### 2. Chạy API server
```bash
cd gradio_demo
python run_api.py
```
hoặc
```bash
cd gradio_demo
python api.py
```

Server sẽ chạy tại: `http://localhost:8000`

## 📖 API Documentation

Khi server đã chạy, bạn có thể xem tài liệu API tại:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## 🎯 API Endpoints

### 1. Health Check
```
GET /health
```
Kiểm tra trạng thái API server.

**Response:**
```json
{
  "status": "healthy",
  "device": "cuda:0"
}
```

### 2. Try-on
```
POST /try-on
```
Thực hiện virtual try-on.

**Parameters (Form Data):**
- `human_image` (file, required): Ảnh người (JPG, PNG)
- `garment_image` (file, required): Ảnh quần áo (JPG, PNG)  
- `garment_description` (string, required): Mô tả quần áo
- `mask_image` (file, optional): Ảnh mask tùy chọn
- `use_auto_mask` (boolean, default=true): Sử dụng auto mask
- `use_auto_crop` (boolean, default=false): Sử dụng auto crop
- `denoise_steps` (integer, default=30): Số bước denoise (20-40)
- `seed` (integer, default=42): Seed cho random generator (-1 to 2147483647)

**Response:**
```json
{
  "success": true,
  "result_image": "base64_encoded_image",
  "mask_image": "base64_encoded_mask",
  "message": "Try-on completed successfully"
}
```

## 💻 Sử dụng từ Python

### Ví dụ cơ bản:
```python
import requests

# Gọi API
files = {
    'human_image': open('human.jpg', 'rb'),
    'garment_image': open('garment.jpg', 'rb')
}
data = {
    'garment_description': 'white t-shirt',
    'use_auto_mask': True,
    'denoise_steps': 30,
    'seed': 42
}

response = requests.post('http://localhost:8000/try-on', files=files, data=data)
result = response.json()

if result['success']:
    # Lưu ảnh kết quả
    import base64
    from PIL import Image
    import io
    
    image_data = base64.b64decode(result['result_image'])
    image = Image.open(io.BytesIO(image_data))
    image.save('result.png')
```

### Chạy demo client:
```bash
cd gradio_demo
python client_example.py
```

## 📱 Sử dụng từ JavaScript/Web

### Ví dụ với HTML form:
```html
<form id="tryonForm" enctype="multipart/form-data">
    <input type="file" name="human_image" accept="image/*" required>
    <input type="file" name="garment_image" accept="image/*" required>
    <input type="text" name="garment_description" placeholder="Mô tả quần áo" required>
    <input type="checkbox" name="use_auto_mask" checked>
    <input type="number" name="denoise_steps" value="30" min="20" max="40">
    <button type="submit">Try-on</button>
</form>

<script>
document.getElementById('tryonForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const formData = new FormData(e.target);
    
    try {
        const response = await fetch('http://localhost:8000/try-on', {
            method: 'POST',
            body: formData
        });
        
        const result = await response.json();
        
        if (result.success) {
            // Hiển thị ảnh kết quả
            const img = document.createElement('img');
            img.src = 'data:image/png;base64,' + result.result_image;
            document.body.appendChild(img);
        }
    } catch (error) {
        console.error('Error:', error);
    }
});
</script>
```

### Ví dụ với fetch API:
```javascript
async function tryOnAPI(humanFile, garmentFile, description) {
    const formData = new FormData();
    formData.append('human_image', humanFile);
    formData.append('garment_image', garmentFile);
    formData.append('garment_description', description);
    formData.append('use_auto_mask', true);
    formData.append('denoise_steps', 30);
    formData.append('seed', 42);
    
    const response = await fetch('http://localhost:8000/try-on', {
        method: 'POST',
        body: formData
    });
    
    return await response.json();
}
```

## 🛠 Troubleshooting

### Lỗi thường gặp:

1. **Models not found**: Đảm bảo đã tải về các model weights
2. **CUDA out of memory**: Giảm batch size hoặc dùng CPU
3. **Port already in use**: Thay đổi port trong `run_api.py`
4. **File too large**: Giảm kích thước ảnh input

### Logs và debugging:
Server sẽ in ra logs chi tiết. Kiểm tra terminal để xem thông tin debug.

## ⚙️ Cấu hình

Bạn có thể thay đổi các cấu hình trong file `api.py`:
- Port: Thay đổi trong `uvicorn.run()`
- Device: Tự động detect CUDA/CPU
- Model path: Thay đổi `base_path`
- CORS settings: Cấu hình trong `CORSMiddleware`

## 🔐 Security Notes

- API này được thiết kế cho development/testing
- Cho production, nên thêm authentication
- Giới hạn file size và rate limiting
- Validate input files kỹ hơn 