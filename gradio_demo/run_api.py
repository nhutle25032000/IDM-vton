#!/usr/bin/env python3
"""
Script để chạy IDM-VTON API Server
"""
import uvicorn
import os
import sys

# Thêm thư mục gốc vào Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if __name__ == "__main__":
    print("🚀 Starting IDM-VTON API Server...")
    print("📖 API Documentation sẽ có tại: http://localhost:8000/docs")
    print("🏥 Health check tại: http://localhost:8000/health")
    print("🎯 Try-on endpoint tại: http://localhost:8000/try-on")
    
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=False,  # Tắt reload vì model loading mất thời gian
        log_level="info"
    ) 