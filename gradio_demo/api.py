import sys
sys.path.append('./')
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
import io
import base64
import json
from typing import Optional
import uvicorn

from src.tryon_pipeline import StableDiffusionXLInpaintPipeline as TryonPipeline
from src.unet_hacked_garmnet import UNet2DConditionModel as UNet2DConditionModel_ref
from src.unet_hacked_tryon import UNet2DConditionModel
from transformers import (
    CLIPImageProcessor,
    CLIPVisionModelWithProjection,
    CLIPTextModel,
    CLIPTextModelWithProjection,
)
from diffusers import DDPMScheduler,AutoencoderKL
from typing import List

import torch
import os
from transformers import AutoTokenizer
import numpy as np
from utils_mask import get_mask_location
from torchvision import transforms
import apply_net
from preprocess.humanparsing.run_parsing import Parsing
from preprocess.openpose.run_openpose import OpenPose
from detectron2.data.detection_utils import convert_PIL_to_numpy,_apply_exif_orientation
from torchvision.transforms.functional import to_pil_image

# Khởi tạo FastAPI app
app = FastAPI(title="IDM-VTON API", description="Virtual Try-on API", version="1.0.0")

# CORS middleware để cho phép gọi từ frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

def pil_to_binary_mask(pil_image, threshold=0):
    np_image = np.array(pil_image)
    grayscale_image = Image.fromarray(np_image).convert("L")
    binary_mask = np.array(grayscale_image) > threshold
    mask = np.zeros(binary_mask.shape, dtype=np.uint8)
    for i in range(binary_mask.shape[0]):
        for j in range(binary_mask.shape[1]):
            if binary_mask[i,j] == True :
                mask[i,j] = 1
    mask = (mask*255).astype(np.uint8)
    output_mask = Image.fromarray(mask)
    return output_mask

def image_to_base64(image: Image.Image) -> str:
    """Chuyển đổi PIL Image thành base64 string"""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str

def base64_to_image(base64_str: str) -> Image.Image:
    """Chuyển đổi base64 string thành PIL Image"""
    image_data = base64.b64decode(base64_str)
    image = Image.open(io.BytesIO(image_data))
    return image

# Khởi tạo models (chỉ một lần khi start server)
base_path = 'yisol/IDM-VTON'

print("Loading models...")

unet = UNet2DConditionModel.from_pretrained(
    base_path,
    subfolder="unet",
    torch_dtype=torch.float16,
)
unet.requires_grad_(False)

tokenizer_one = AutoTokenizer.from_pretrained(
    base_path,
    subfolder="tokenizer",
    revision=None,
    use_fast=False,
)
tokenizer_two = AutoTokenizer.from_pretrained(
    base_path,
    subfolder="tokenizer_2",
    revision=None,
    use_fast=False,
)
noise_scheduler = DDPMScheduler.from_pretrained(base_path, subfolder="scheduler")

text_encoder_one = CLIPTextModel.from_pretrained(
    base_path,
    subfolder="text_encoder",
    torch_dtype=torch.float16,
)
text_encoder_two = CLIPTextModelWithProjection.from_pretrained(
    base_path,
    subfolder="text_encoder_2",
    torch_dtype=torch.float16,
)
image_encoder = CLIPVisionModelWithProjection.from_pretrained(
    base_path,
    subfolder="image_encoder",
    torch_dtype=torch.float16,
)
vae = AutoencoderKL.from_pretrained(base_path,
                                    subfolder="vae",
                                    torch_dtype=torch.float16,
)

UNet_Encoder = UNet2DConditionModel_ref.from_pretrained(
    base_path,
    subfolder="unet_encoder",
    torch_dtype=torch.float16,
)

parsing_model = Parsing(0)
openpose_model = OpenPose(0)

UNet_Encoder.requires_grad_(False)
image_encoder.requires_grad_(False)
vae.requires_grad_(False)
unet.requires_grad_(False)
text_encoder_one.requires_grad_(False)
text_encoder_two.requires_grad_(False)

tensor_transfrom = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])

pipe = TryonPipeline.from_pretrained(
    base_path,
    unet=unet,
    vae=vae,
    feature_extractor=CLIPImageProcessor(),
    text_encoder=text_encoder_one,
    text_encoder_2=text_encoder_two,
    tokenizer=tokenizer_one,
    tokenizer_2=tokenizer_two,
    scheduler=noise_scheduler,
    image_encoder=image_encoder,
    torch_dtype=torch.float16,
)
pipe.unet_encoder = UNet_Encoder

print("Models loaded successfully!")

@app.get("/")
async def root():
    return {"message": "IDM-VTON API is running!"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "device": device}

@app.post("/try-on")
async def try_on_api(
    human_image: UploadFile = File(..., description="Ảnh người (JPG, PNG)"),
    garment_image: UploadFile = File(..., description="Ảnh quần áo (JPG, PNG)"),
    garment_description: str = Form(..., description="Mô tả quần áo"),
    mask_image: Optional[UploadFile] = File(None, description="Ảnh mask tùy chọn (nếu không dùng auto mask)"),
    use_auto_mask: bool = Form(True, description="Sử dụng auto mask"),
    use_auto_crop: bool = Form(False, description="Sử dụng auto crop và resize"),
    denoise_steps: int = Form(30, description="Số bước denoise (20-40)"),
    seed: int = Form(42, description="Seed cho random generator")
):
    try:
        # Validate parameters
        if denoise_steps < 20 or denoise_steps > 40:
            raise HTTPException(status_code=400, detail="denoise_steps phải trong khoảng 20-40")
        
        if seed < -1 or seed > 2147483647:
            raise HTTPException(status_code=400, detail="seed phải trong khoảng -1 đến 2147483647")
        
        # Đọc ảnh từ upload files
        human_img_bytes = await human_image.read()
        garment_img_bytes = await garment_image.read()
        
        human_img = Image.open(io.BytesIO(human_img_bytes)).convert("RGB")
        garment_img = Image.open(io.BytesIO(garment_img_bytes)).convert("RGB")
        
        # Chuẩn bị dict cho human image (giống như ImageEditor trong Gradio)
        human_dict = {
            "background": human_img,
            "layers": [Image.open(io.BytesIO(await mask_image.read())).convert("RGB")] if mask_image else [None],
            "composite": None
        }
        
        # Gọi hàm try-on
        result_image, mask_image_result = await process_tryon(
            human_dict, 
            garment_img, 
            garment_description, 
            use_auto_mask, 
            use_auto_crop, 
            denoise_steps, 
            seed
        )
        
        # Chuyển đổi kết quả thành base64
        result_base64 = image_to_base64(result_image)
        mask_base64 = image_to_base64(mask_image_result)
        
        return {
            "success": True,
            "result_image": result_base64,
            "mask_image": mask_base64,
            "message": "Try-on completed successfully"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi xử lý: {str(e)}")

async def process_tryon(dict_img, garm_img, garment_des, is_checked, is_checked_crop, denoise_steps, seed):
    """
    Hàm xử lý try-on (async version của start_tryon)
    """
    
    openpose_model.preprocessor.body_estimation.model.to(device)
    pipe.to(device)
    pipe.unet_encoder.to(device)

    garm_img = garm_img.convert("RGB").resize((768,1024))
    human_img_orig = dict_img["background"].convert("RGB")    
    
    if is_checked_crop:
        width, height = human_img_orig.size
        target_width = int(min(width, height * (3 / 4)))
        target_height = int(min(height, width * (4 / 3)))
        left = (width - target_width) / 2
        top = (height - target_height) / 2
        right = (width + target_width) / 2
        bottom = (height + target_height) / 2
        cropped_img = human_img_orig.crop((left, top, right, bottom))
        crop_size = cropped_img.size
        human_img = cropped_img.resize((768,1024))
    else:
        human_img = human_img_orig.resize((768,1024))

    if is_checked:
        keypoints = openpose_model(human_img.resize((384,512)))
        model_parse, _ = parsing_model(human_img.resize((384,512)))
        mask, mask_gray = get_mask_location('hd', "upper_body", model_parse, keypoints)
        mask = mask.resize((768,1024))
    else:
        if dict_img['layers'][0] is not None:
            mask = pil_to_binary_mask(dict_img['layers'][0].convert("RGB").resize((768, 1024)))
        else:
            # Nếu không có mask được cung cấp, sử dụng auto mask
            keypoints = openpose_model(human_img.resize((384,512)))
            model_parse, _ = parsing_model(human_img.resize((384,512)))
            mask, mask_gray = get_mask_location('hd', "upper_body", model_parse, keypoints)
            mask = mask.resize((768,1024))
    
    mask_gray = (1-transforms.ToTensor()(mask)) * tensor_transfrom(human_img)
    mask_gray = to_pil_image((mask_gray+1.0)/2.0)

    human_img_arg = _apply_exif_orientation(human_img.resize((384,512)))
    human_img_arg = convert_PIL_to_numpy(human_img_arg, format="BGR")

    args = apply_net.create_argument_parser().parse_args(('show', './configs/densepose_rcnn_R_50_FPN_s1x.yaml', './ckpt/densepose/model_final_162be9.pkl', 'dp_segm', '-v', '--opts', 'MODEL.DEVICE', 'cuda'))
    pose_img = args.func(args, human_img_arg)    
    pose_img = pose_img[:,:,::-1]    
    pose_img = Image.fromarray(pose_img).resize((768,1024))
    
    with torch.no_grad():
        with torch.cuda.amp.autocast():
            with torch.no_grad():
                prompt = "model is wearing " + garment_des
                negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"
                with torch.inference_mode():
                    (
                        prompt_embeds,
                        negative_prompt_embeds,
                        pooled_prompt_embeds,
                        negative_pooled_prompt_embeds,
                    ) = pipe.encode_prompt(
                        prompt,
                        num_images_per_prompt=1,
                        do_classifier_free_guidance=True,
                        negative_prompt=negative_prompt,
                    )
                                    
                    prompt = "a photo of " + garment_des
                    negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"
                    if not isinstance(prompt, List):
                        prompt = [prompt] * 1
                    if not isinstance(negative_prompt, List):
                        negative_prompt = [negative_prompt] * 1
                    with torch.inference_mode():
                        (
                            prompt_embeds_c,
                            _,
                            _,
                            _,
                        ) = pipe.encode_prompt(
                            prompt,
                            num_images_per_prompt=1,
                            do_classifier_free_guidance=False,
                            negative_prompt=negative_prompt,
                        )

                    pose_img = tensor_transfrom(pose_img).unsqueeze(0).to(device,torch.float16)
                    garm_tensor = tensor_transfrom(garm_img).unsqueeze(0).to(device,torch.float16)
                    generator = torch.Generator(device).manual_seed(seed) if seed is not None else None
                    images = pipe(
                        prompt_embeds=prompt_embeds.to(device,torch.float16),
                        negative_prompt_embeds=negative_prompt_embeds.to(device,torch.float16),
                        pooled_prompt_embeds=pooled_prompt_embeds.to(device,torch.float16),
                        negative_pooled_prompt_embeds=negative_pooled_prompt_embeds.to(device,torch.float16),
                        num_inference_steps=denoise_steps,
                        generator=generator,
                        strength=1.0,
                        pose_img=pose_img.to(device,torch.float16),
                        text_embeds_cloth=prompt_embeds_c.to(device,torch.float16),
                        cloth=garm_tensor.to(device,torch.float16),
                        mask_image=mask,
                        image=human_img, 
                        height=1024,
                        width=768,
                        ip_adapter_image=garm_img.resize((768,1024)),
                        guidance_scale=2.0,
                    )[0]

    if is_checked_crop:
        out_img = images[0].resize(crop_size)        
        human_img_orig.paste(out_img, (int(left), int(top)))    
        return human_img_orig, mask_gray
    else:
        return images[0], mask_gray

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 