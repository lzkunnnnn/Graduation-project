import os.path
import tempfile
import uuid
from pathlib import Path
from tempfile import NamedTemporaryFile

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from starlette.responses import StreamingResponse, FileResponse

import Comparison
import tsuyoi


app = FastAPI()

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有源
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# @app.post("/videopro/")
# async def video_pro(video: UploadFile = File(...)):
#     output_filename = f"processed_{uuid.uuid4().hex}.mp4"
#     temp_video = None
#     output_path = None
#
#     try:
#         with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
#             content = await video.read()
#             temp_video.write(content)
#             temp_path = temp_video.name
#
#         gen = tsuyoi.process_video(
#             model_path="yolo11n.pt",
#             video_path=temp_path,
#             save_dir="../output"
#         )
#
#         output_dir = os.path.abspath("../output")
#         os.makedirs(output_dir, exist_ok=True)
#         output_path = os.path.join(output_dir, output_filename)
#
#         tsuyoi.combined_processing(gen, output_path)
#
#         if not os.path.exists(output_path):
#             raise FileNotFoundError("处理后的视频文件未生成")
#
#         return FileResponse(
#             path=output_path,
#             media_type="video/mp4",
#             filename=output_filename
#         )
#
#     except Exception as e:
#         return {"error": str(e)}


@app.post("/upload-img")
async def upload_img(file: UploadFile = File(...)):
    try:
        with NamedTemporaryFile(delete=False, suffix=".jpg") as temp:
            content = await file.read()
            temp.write(content)
            temp_path = temp.name

        most_similar_image_path, _ = Comparison.find_most_similar_image(
            temp_path,
            "../output/video"
        )

        current_dir = os.path.dirname(os.path.abspath(most_similar_image_path))

        frame_file = next(Path(current_dir).glob("frame*"), None)

        if not frame_file:
            return {"error": "未找到frame图片"}

        def file_stream():
            with open(frame_file, "rb") as f:
                yield from f

        return StreamingResponse(
            file_stream(),
            media_type="image/jpeg" if str(frame_file).lower().endswith('.jpg') else "image/png"
        )

    except Exception as e:
        return {"error": str(e)}


if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8000)
