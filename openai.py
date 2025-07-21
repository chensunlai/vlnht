from __future__ import annotations

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import asyncio
import uuid
from io import BytesIO
from pathlib import Path
from typing import Dict, List
import numpy as np
import torch
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel
from PIL import Image
import copy

from llava.constants import IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates, SeparatorStyle
from llava.mm_utils import (
    KeywordsStoppingCriteria,
    get_model_name_from_path,
    process_images,
    tokenizer_image_token,
)
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init

MODEL_PATH = "model/navila-llama3-8b-8f"  

def load_navila_model():
    disable_torch_init()  
    model_name = get_model_name_from_path(MODEL_PATH)
    tokenizer, model, image_processor, _ = load_pretrained_model(MODEL_PATH, model_name, None)
    model.eval()
    return tokenizer, model, image_processor


TOKENIZER, MODEL, IMAGE_PROCESSOR = load_navila_model()
DEVICE = MODEL.device  

TASKS: Dict[str, List[bytes]] = {}
TASK_LOCK = asyncio.Lock()  

app = FastAPI()


class CreateTaskResponse(BaseModel):
    task_id: str


class AddImageResponse(BaseModel):
    task_id: str
    image_count: int


class InferRequest(BaseModel):
    instruction: str


class InferResponse(BaseModel):
    task_id: str
    result: str
    image_count: int

def bytes_to_pil(data: bytes) -> Image.Image:
    try:
        return Image.open(BytesIO(data)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"无效图像: {e}")
    
def sample_and_pad_images(frames, num_frames=8):
    (width, height) = frames[-1].size
    while len(frames) < num_frames:
        frames.insert(0, Image.new("RGB", (width, height), color=(0, 0, 0)))

    latest_frame = frames[-1]
    sampled_indices = np.linspace(0, len(frames) - 1, num=num_frames - 1, endpoint=False, dtype=int)
    sampled_frames = [frames[i] for i in sampled_indices] + [latest_frame]

    return sampled_frames

def run_inference(images: List[bytes], instruction: str) -> str:
    if not images:
        raise ValueError("任务图像列表为空，无法推理。")

    pil_images = [bytes_to_pil(b) for b in images]
    pil_images = sample_and_pad_images(pil_images)

    images_tensor = process_images(pil_images, IMAGE_PROCESSOR, MODEL.config).to(DEVICE, dtype=torch.float16)

    conv = conv_templates["llama_3"].copy()
    img_token = "<image>\n"
    user_prompt = (
        f"Imagine you are a robot programmed for navigation tasks. You have been given a video "
        f'of historical observations {img_token * max(len(pil_images) - 1, 0)}, and current observation <image>\n. Your assigned task is: "{instruction}" '
        f"Analyze this series of images to decide your next action, which could be turning left or right by a specific "
        f"degree, moving forward a certain distance, or stop if the task is completed."
    )
    
    conv.append_message(conv.roles[0], user_prompt)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(prompt, TOKENIZER, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(DEVICE)

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    stopping = KeywordsStoppingCriteria([stop_str], TOKENIZER, input_ids)

    with torch.inference_mode():
        output_ids = MODEL.generate(
            input_ids,
            images=images_tensor,
            do_sample=False,
            temperature=0.0,
            max_new_tokens=1024,
            use_cache=True,
            stopping_criteria=[stopping],
            pad_token_id=TOKENIZER.eos_token_id,
        )

    text = TOKENIZER.decode(output_ids[0], skip_special_tokens=True).strip()
    if text.endswith(stop_str):
        text = text[:-len(stop_str)].strip()
    return text

@app.post("/api/tasks", response_model=CreateTaskResponse, summary="创建任务")
async def create_task():
    task_id = uuid.uuid4().hex
    async with TASK_LOCK:
        TASKS[task_id] = []
    return CreateTaskResponse(task_id=task_id)


@app.post("/api/tasks/{task_id}/images", response_model=AddImageResponse, summary="上传图像")
async def add_image(task_id: str, file: UploadFile = File(...)):
    if file.content_type not in {"image/jpeg", "image/png", "image/jpg"}:
        raise HTTPException(status_code=400, detail="仅支持 JPEG/PNG 格式")
    data = await file.read()

    async with TASK_LOCK:
        if task_id not in TASKS:
            raise HTTPException(status_code=404, detail="任务不存在")
        TASKS[task_id].append(data)
        count = len(TASKS[task_id])

    save_dir = Path("runs") / task_id      # images/{task_id}
    save_dir.mkdir(parents=True, exist_ok=True)

    # 根据 MIME 类型决定扩展名
    suffix = ".png" if file.content_type == "image/png" else ".jpg"
    save_path = save_dir / f"{count:04d}{suffix}"

    # try:
    #     with save_path.open("wb") as f_out:
    #         f_out.write(data)
    #     with (Path("runs") / f"realtime{suffix}").open("wb") as f_out:
    #         f_out.write(data)
            
    #     with open(os.path.join(save_dir, "log"), "a+") as fp:
    #         fp.write(f"{count:04d}{suffix}" + "\n") 
    # except Exception as e:
    #     raise HTTPException(status_code=500, detail=f"保存文件失败: {e}")
        
    return AddImageResponse(task_id=task_id, image_count=count)


@app.post("/api/tasks/{task_id}/infer", response_model=InferResponse, summary="执行推理")
async def inference(task_id: str, body: InferRequest):
    async with TASK_LOCK:
        if task_id not in TASKS:
            raise HTTPException(status_code=404, detail="任务不存在")
        images = list(TASKS[task_id]) 
    if not images:
        raise HTTPException(status_code=400, detail="该任务尚未上传任何图像")

    def _run():
        return run_inference(images, body.instruction)

    loop = asyncio.get_event_loop()
    result: str = await loop.run_in_executor(None, _run)
    save_dir = Path("runs") / task_id
    with open(os.path.join(save_dir, "log"), "a+") as fp:
        fp.write(body.instruction + " | " + result + "\n") 
    return InferResponse(task_id=task_id, result=result, image_count=len(images))

if __name__ == "__main__":
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="Run NaVILA FastAPI service programmatically")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind")
    parser.add_argument("--port", type=int, default=18880, help="Port to listen on")
    args = parser.parse_args()

    uvicorn.run("openai:app", host=args.host, port=args.port, reload=False)