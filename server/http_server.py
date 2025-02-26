#coding:utf-8
from fastapi import FastAPI
from engine.engine import inference_engine
from pydantic import BaseModel
from http_server_config import ServerConfig, ModelConfig

app = FastAPI()
server_config = ServerConfig()
model_config = ModelConfig()
engine = inference_engine(cfg_path=model_config.cfg_path,
                          ckpt_path=model_config.ckpt_path,
                          device_type=model_config.device_type,
                          gpus=model_config.gpus)

class InputData(BaseModel):
    data: list

@app.put("/inference")
async def inference(input_data: InputData):
    if not isinstance(input_data.data, list):
        return {"error": "Input data must be a list"}
    predictions = engine.inference(input_data.data)
    return {"result": predictions.tolist()}


if __name__ == '__main__':
    import uvicorn
    uvicorn.run('server.http_server:app', host=server_config.host, port=server_config.port, reload=True)