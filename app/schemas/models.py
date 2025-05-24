from pydantic import BaseModel, Field
from typing import Annotated

class RulesUsed(BaseModel):
    rules_used: dict[str, bool]

class RuleWeights(BaseModel):
    rule_weights: dict[str, float]

class Range(BaseModel):
    lower: float
    upper: float

class Defenses(BaseModel):
    crouching: float
    standing: float

class ContextParameters(BaseModel):
    attacks_landed: Range
    current_hp: Annotated[float, Field(strict=True, ge=0, le=200)]
    defenses: Defenses
    lower_hits: float
    upper_hits: float
