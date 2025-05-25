from fastapi import FastAPI
from .lstm.router import router
from .lstm.lstm import load_model
from contextlib import asynccontextmanager

app = FastAPI(
    title="LSTM Server",
)

app.include_router(router)

