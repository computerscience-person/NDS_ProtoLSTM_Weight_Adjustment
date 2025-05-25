from fastapi import APIRouter, HTTPException, status

from ..schemas.models import ContextParameters
from .model import router as model_0_0_5
from .lstm import predict as predict_model, buffer_input, get_buffer

router = APIRouter(
    prefix="/api/v162",
    tags=['nds', 'lstm']
)

router.include_router(model_0_0_5.router)


@router.post("/predict")
async def predict():
    print(predict_model(get_buffer()).tolist())
    return { "weight_adjustments": predict_model(get_buffer()).tolist() }
    

@router.post("/metrics_in", status_code=201)
async def metrics_in_ok(model_input: ContextParameters):
    print(model_input)
    buffer_input(model_input)
