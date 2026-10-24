import io
import os
import urllib.request
from typing import Optional
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import onnxruntime as ort
from PIL import Image

app = FastAPI(title="Faster Remover Pro 10X API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = "/tmp/rmbg14_quant.onnx"
MODEL_URL = "https://huggingface.co/briaai/RMBG-1.4/resolve/main/onnx/model_quantized.onnx"
session = None

def get_session():
    global session
    if session is None:
        if not os.path.exists(MODEL_PATH):
            req = urllib.request.Request(
                MODEL_URL, 
                headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
            )
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
    return {"status": "online", "model": "RMBG-1.4 10X Ultra HD Ready"}

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

        # ১. ১০২৪x১০২৪ সাইজে অপটিমাল রিসাইজ
        img_resized = img.resize((1024, 1024), Image.Resampling.BILINEAR)

        # ২. অফিসিয়াল প্রি-প্রসেসিং: (pixel / 255.0 - 0.5) / 1.0
        img_data = np.array(img_resized).astype(np.float32) / 255.0
        img_data = (img_data - 0.5) / 1.0

        img_data = np.transpose(img_data, (2, 0, 1))
        img_data = np.expand_dims(img_data, axis=0)

        # ৩. মডেল ইনফারেন্স
        ort_sess = get_session()
        input_name = ort_sess.get_inputs()[0].name
        outputs = ort_sess.run(None, {input_name: img_data})
        
        pred = outputs[0]
        while len(pred.shape) > 2:
            pred = pred[0]

        # ৪. র' স্কেলিং
        mi = float(np.min(pred))
        ma = float(np.max(pred))
        if ma > mi:
            norm = (pred - mi) / (ma - mi)
        else:
            norm = pred

        # ৫. নয়েজ কিলার ও ব্যাকগ্রাউন্ড ক্লিনআপ ফিল্টার
        # ৩৫%-এর নিচে থাকা সব অবশিষ্টাংশ, সাদা দাগ ও লো-কোয়ালিটি নয়েজ ১০০% ভ্যানিশ
        # ৬৫%-এর ওপরে থাকা মূল মানুষ/সাবজেক্ট ১০০% সলিড
        norm = np.clip((norm - 0.35) / (0.65 - 0.35), 0.0, 1.0)
        norm = norm * norm * (3.0 - 2.0 * norm)  # মসৃণ হারমাইট কার্ভ

        # ৬. আল্ট্রা-এইচডি মাস্কিং
        mask_1024 = Image.fromarray((norm * 255.0).astype(np.uint8), mode="L")
        mask = mask_1024.resize((orig_w, orig_h), Image.Resampling.LANCZOS)

        # ৭. ট্রান্সপারেন্ট পিএনজি তৈরি
        result = img.copy()
        result.putalpha(mask)

        out_io = io.BytesIO()
        result.save(out_io, format="PNG")
        return Response(content=out_io.getvalue(), media_type="image/png")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
