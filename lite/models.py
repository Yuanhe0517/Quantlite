from typing import Literal
from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(extra='forbid', str_strip_whitespace=True)


class Research(StrictModel):
    direction: str = Field(min_length=1)
    market: Literal['csi300', 'all_a'] = 'csi300'
    horizon_days: int = Field(default=21, ge=1, le=252)
    fields: list[Literal['$open', '$high', '$low', '$close', '$volume', '$return']] = Field(
        default_factory=lambda: ['$open', '$high', '$low', '$close', '$volume', '$return'], min_length=1)


class LLM(StrictModel):
    model: str = Field(default='gpt-6.1-terra', min_length=1)
    reasoning_effort: Literal['low', 'medium', 'high'] = 'low'
    max_output_tokens: int = Field(default=8000, ge=1000, le=32000)
    timeout_seconds: int = Field(default=180, ge=1, le=600)


class LoopConfig(StrictModel):
    max_rounds: int = Field(default=4, ge=1, le=30)
    factors_per_round: Literal[2] = 2
    start: Literal['2020-01-01'] = '2020-01-01'
    end: Literal['2023-12-31'] = '2023-12-31'
    warmup_start: Literal['2008-12-29'] = '2008-12-29'
    provider_uri: str = '../QuantaAlpha/data/qlib/cn_data'


class Config(StrictModel):
    research: Research
    loop: LoopConfig = Field(default_factory=LoopConfig)
    llm: LLM = Field(default_factory=LLM)


class Hypothesis(StrictModel):
    hypothesis: str = Field(min_length=1)
    concise_knowledge: str = Field(min_length=1)
    concise_observation: str = Field(min_length=1)
    concise_justification: str = Field(min_length=1)
    concise_specification: str = Field(min_length=1)


class Factor(StrictModel):
    # Optional explanatory metadata returned by the original revision prompts.
    concise_observation: str | None = None
    concise_justification: str | None = None
    concise_specification: str | None = None
    description: str = Field(min_length=1)
    variables: dict[str, str] = Field(min_length=1)
    formulation: str = Field(min_length=1)
    expression: str = Field(min_length=1)


def validate_factors(data: dict) -> dict:
    if not isinstance(data, dict) or not 2 <= len(data) <= 3:
        raise ValueError('Expected 2-3 factors for one hypothesis, as in original prompt')
    result = {}
    for name, value in data.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError('Factor names must be nonempty strings')
        result[name] = Factor.model_validate(value).model_dump(exclude_none=True)
    return result
