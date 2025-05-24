from fastapi import APIRouter
from ..schemas.models import ContextParameters

router = APIRouter(
    prefix="/api/v162",
    tags=['nds', 'lstm']
)

@router.post("/hello")
async def hello():
    return "hello"

@router.post("/predict")
async def predict(model_input: ContextParameters):
    return "prediction"

@router.post("/metrics_in")
async def metrics_in(model_input: ContextParameters):
    return "prediction"
