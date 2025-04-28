from fastapi import FastAPI
from .schemas.models import RuleWeights, RulesUsed
from .typing_utils import convert_dict_to_sorted_list

app = FastAPI(
    title="LSTM Server"
)

@app.get("/")
async def root():
    return {"message": "Hello World!"}

@app.post("/api/v0/update_weights")
async def update_weights(rules_used: RulesUsed) -> RuleWeights :
    rules, was_used = convert_dict_to_sorted_list(rules_used)
    weights: RuleWeights = RuleWeights(rule_weights={"2": 0.8})
    return weights

@app.post("/api/v0/model_ready")
async def model_ready() -> str:
    return "Model ready."
