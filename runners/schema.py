"""Prompt JSON schema + loader. Source of truth: docs/PRD.md §6."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

TaskCategory = Literal[
    "customer_support",
    "rag_qa",
    "extraction",
    "summarisation",
    "reasoning",
]
Complexity = Literal["easy", "medium", "hard"]


class PromptInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    system: str = Field(min_length=1)
    user: str = Field(min_length=1)


class Tier1Scoring(BaseModel):
    model_config = ConfigDict(extra="forbid")
    expected: dict[str, Any]


class Tier2Scoring(BaseModel):
    model_config = ConfigDict(extra="forbid")
    criteria: str = Field(min_length=1)


class Scoring(BaseModel):
    model_config = ConfigDict(extra="forbid")
    tier_1_deterministic: Tier1Scoring | None = None
    tier_2_judge: Tier2Scoring | None = None

    @model_validator(mode="after")
    def at_least_one_tier(self) -> "Scoring":
        if self.tier_1_deterministic is None and self.tier_2_judge is None:
            raise ValueError("scoring must define at least one of tier_1_deterministic or tier_2_judge")
        return self


class PromptMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")
    input_tokens_approx: int = Field(ge=0)
    notes: str | None = None


class Prompt(BaseModel):
    model_config = ConfigDict(extra="forbid")
    prompt_id: str = Field(min_length=1)
    task_category: TaskCategory
    complexity: Complexity
    input: PromptInput
    scoring: Scoring
    metadata: PromptMetadata


def load_prompts(path: str | Path) -> list[Prompt]:
    """Load and validate a prompts JSON file. Raises pydantic.ValidationError on bad shape."""
    raw = json.loads(Path(path).read_text())
    if not isinstance(raw, list):
        raise ValueError(f"{path}: top-level JSON must be an array of prompt objects")
    prompts = [Prompt.model_validate(p) for p in raw]
    ids = [p.prompt_id for p in prompts]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{path}: duplicate prompt_id values")
    return prompts
