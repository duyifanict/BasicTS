# coding: utf-8
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict

class BasicConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix='bts_server_')

class ServerConfig(BasicConfig):
    host: str = '0.0.0.0'
    port: int = 5000

class ModelConfig(BasicConfig):
    cfg_path: str = 'baselines\\STID\\PEMS04.py'
    ckpt_path: str = 'C:\\Users\\D\\code\\github-duyifanict\\BasicTS\\checkpoints\\STID\\PEMS04_100_12_12\\5684a53d44870276f5fb6522f26cf035\\STID_best_val_MAE.pt'
    device_type: str = 'cpu'
    gpus: Optional[str] = None
    