from pathlib import Path
from fastapi import APIRouter
from ...schemas.models import ContextParameters
from .. import lstm

router = APIRouter(
    prefix="/model_0_0_5",
    tags=['nds', 'lstm']
)

@router.post("/load_model", status_code=201)
async def load_model_0_0_5():
    lstm.load_model(Path("./app/lstm/rule_adjustment_model_0_0_5.keras"))
