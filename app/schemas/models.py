from pydantic import BaseModel

class RulesUsed(BaseModel):
    rules_used: dict[str, bool]

class RuleWeights(BaseModel):
    rule_weights: dict[str, float]

