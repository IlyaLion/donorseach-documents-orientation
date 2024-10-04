from fastapi import FastAPI, Response, Request, UploadFile
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

import uvicorn
import argparse
import cv2

from PIL import Image
from io import BytesIO
import numpy as np

from rotator import Rotator

app = FastAPI()
rotator = Rotator(
    backbone_name='convnext_femto.d1_in1k',
    model_weights_filename='weights/convnext_femto.d1_in1k_weights.ckpt',
    image_size=128
    )

#app.mount("/tmp", StaticFiles(directory="tmp"), name='images')
templates = Jinja2Templates(directory="templates")

@app.head("/ping")
@app.get("/ping")
async def ping(response: Response):
    response.headers["status"] = "OK"

@app.get("/")
async def root(request: Request):
    return templates.TemplateResponse("start_form.html",
                                      {"request": request})

@app.post("/predict")
async def predict(file: UploadFile):  #request: Request

    image = np.array(Image.open(BytesIO(file.file.read())).convert("RGB"))
    
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    rotated_image = rotator.rotate(image)
    rotated_image = Image.fromarray(rotated_image)
    rotated_image.save('temp.png')
    return FileResponse('temp.png')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default=8000, type=int, dest="port")
    parser.add_argument("--host", default="127.0.0.1", type=str, dest="host")
    args = vars(parser.parse_args())

    uvicorn.run(app, **args)
    