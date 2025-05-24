from fastapi import FastAPI
from .lstm.router import router
from .lstm.lstm import load_model
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield

app = FastAPI(
    title="LSTM Server",
    lifespan=lifespan
)

app.include_router(router)

