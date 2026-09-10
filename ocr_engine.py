"""本地 CPU OCR；字节输入统一转为白底 BGR，避免透明背景和通道问题。"""
import io
import threading

import numpy as np
from PIL import Image, ImageOps


class OCREngine:
    def __init__(self):
        from rapidocr_onnxruntime import RapidOCR
        self.engine = RapidOCR(intra_op_num_threads=2, inter_op_num_threads=2)
        self.lock = threading.Lock()

    def __call__(self, data):
        with Image.open(io.BytesIO(data)) as source:
            if source.width * source.height > 40_000_000:
                raise ValueError('图片超过 4000 万像素，请缩小后上传')
            upright = ImageOps.exif_transpose(source)
            rgba = upright.convert('RGBA')
            canvas = Image.new('RGBA', rgba.size, 'white')
            canvas.alpha_composite(rgba)
            rgb = canvas.convert('RGB')
            rgb.thumbnail((3200, 3200))
            array = np.asarray(rgb)[:, :, ::-1].copy()
        with self.lock:
            result, _ = self.engine(array)
        return '\n'.join(str(row[1]) for row in (result or []) if float(row[2]) >= 0.5)
