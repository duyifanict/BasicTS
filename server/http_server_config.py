# coding: utf-8
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict

class BasicConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix='bts_server_')

class ServerConfig(BasicConfig):
    host: str = '0.0.0.0'
    port: int = 5555 

class ModelConfig(BasicConfig):
    cfg_path: str = 'baselines/STID/PEMS04.py'
    ckpt_path: str = 'checkpoints/STID/PEMS04_100_12_12/40de8ae98d3fb912c552c42197c02428/STID_best_val_MAE.pt'
    device_type: str = 'cpu'
    gpus: Optional[str] = None
    