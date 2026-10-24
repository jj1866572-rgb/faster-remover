import io
import os
import urllib.request
from typing import Optional
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import onnxruntime as ort
from PIL import Image, ImageFilter

app = FastAPI(title="Faster Remover Ultra Studio API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# বিশ্বমানের silueta ফুল-প্রিসিশন FP32 ইঞ্জিন (৪৩.৯ এমবি - নো নয়েজ, নো গ্রিন ডট)
MODEL_PATH = "/tmp/silueta.onnx"
PRIMARY_URL = "https://github.com/danielgatis/rembg/releases/download/v0.0.0/silueta.onnx"
FALLBACK_URL = "https://huggingface.co/martintomov/comfy/resolve/main/rembg/silueta.onnx"

session = None

def get_session():
    global session
    if session is None:
        if not os.path.exists(MODEL_PATH):
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
            try:
                req = urllib.request.Request(PRIMARY_URL, headers=headers)
                with urllib.request.urlopen(req) as resp, open(MODEL_PATH, 'wb') as f:
                    f.write(resp.read())
            except Exception:
                req = urllib.request.Request(FALLBACK_URL, headers=headers)
                with urllib.request.urlopen(req) as resp, open(MODEL_PATH, 'wb') as f:
                    f.write(resp.read())

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 2
        session = ort.InferenceSession(MODEL_PATH, opts, providers=['CPUExecutionProvider'])
    return session

@app.get("/")
@app.get("/api")
@app.get("/api/index.py")
def health():
    return {"status": "online", "model": "silueta Studio Quality FP32 Ready"}

@app.post("/")
@app.post("/api")
@app.post("/remove-bg")
@app.post("/api/remove-bg")
@app.post("/api/index.py")
@app.post("/api/index.py/remove-bg")
async def remove_bg(
    image: Optional[UploadFile] = File(None),
    file: Optional[UploadFile] = File(None)
):
    target = image if image is not None else file
    if target is None:
        raise HTTPException(status_code=400, detail="No image provided")

    try:
        contents = await target.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        orig_w, orig_h = img.size

        # ১. ৩২০x৩২০ ইমেজ প্রি-প্রসেসিং (স্ট্যান্ডার্ড ImageNet নরমালাইজেশন)
        img_resized = img.resize((320, 320), Image.Resampling.BILINEAR)
        img_data = np.array(img_resized).astype(np.float32) / 255.0
        
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_data = (img_data - mean) / std
        img_data = np.transpose(img_data, (2, 0, 1))
        img_data = np.expand_dims(img_data, axis=0)

        # ২. silueta এআই মডেল রান
        ort_sess = get_session()
        input_name = ort_sess.get_inputs()[0].name
        outputs = ort_sess.run(None, {input_name: img_data})
        pred = outputs[0][0, 0]

        # ৩. প্রবাবিলিটি নরমালাইজেশন
        d_min, d_max = pred.min(), pred.max()
        norm = (pred - d_min) / (d_max - d_min + 1e-8)
        norm = np.clip(norm, 0.0, 1.0)

        # ৪. ব্যাকগ্রাউন্ড ক্লিনআপ কার্ভ (পেছনের ক্যাফে, টেবিল ও গাছপালা ১০০% নিশ্চিহ্ন করা)
        # ০.৪০ এর নিচে থাকা সব ব্যাকগ্রাউন্ড পুরোপুরি জিরো
        # ০.৭০ এর ওপরে থাকা মূল মানুষ ১০০% সলিড
        norm = np.clip((norm - 0.40) / (0.70 - 0.40), 0.0, 1.0)
        norm = norm * norm * (3.0 - 2.0 * norm)  # মসৃণ হারমাইট কার্ভ

        # ৫. হাই-রেজুলিউশন মাস্ক আসল সাইজে নেওয়া
        mask_320 = (norm * 255.0).astype(np.uint8)
        mask = Image.fromarray(mask_320, mode="L").resize(
            (orig_w, orig_h), Image.Resampling.LANCZOS
        )

        # ৬. মর্ফোলজিক্যাল স্পেকল কিলার (Opening Filter)
        # এটি যেকোনো ভাসমান সবুজ ডট, পাতা বা নয়েজের অস্তিত্ব সম্পূর্ণ মুছে দেবে
        mask = mask.filter(ImageFilter.MinFilter(3)).filter(ImageFilter.MaxFilter(3))

        # ৭. সাব-পিক্সেল স্মুথনেস (যাতে কোনো খাবরা ভাব না থাকে)
        mask = mask.filter(ImageFilter.GaussianBlur(radius=0.7))

        # ৮. ট্রান্সপারেন্ট পিএনজি তৈরি
        result = img.copy()
        result.putalpha(mask)

        out_io = io.BytesIO()
        result.save(out_io, format="PNG")
        return Response(content=out_io.getvalue(), media_type="image/png")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
