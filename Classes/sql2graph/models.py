"""Pydantic models for SQL2Graph extraction contracts."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator


class ColumnRef(BaseModel):
    table_alias: str | None = None
    column: str
    physical_table: str | None = None

    @field_validator("table_alias")
    def normalize_alias(cls, value: str | None) -> str | None:
        return value.strip() if isinstance(value, str) and value.strip() else None

    @field_validator("column")
    def normalize_column(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("column cannot be empty")
        return cleaned

    def node_id(self) -> str:
        alias = self.table_alias or "unknown"
        return f"{alias}.{self.column}"


class OutputColumn(BaseModel):
    alias: str
    expression: str = ""
    dependencies: list[ColumnRef] = Field(default_factory=list)
    aggregate: bool = False
    window_function: bool = False
    derivation_kind: str | None = None
    literal_values: list[str] = Field(default_factory=list)
    union_branches: list[dict[str, Any]] = Field(default_factory=list)


class FilterSpec(BaseModel):
    clause: str
    condition: str
    columns_used: list[ColumnRef] = Field(default_factory=list)


class JoinSpec(BaseModel):
    type: str
    left_alias: str
    right_alias: str
    condition: str
    join_columns: list[ColumnRef]

    @field_validator("join_columns")
    def validate_join_pair(cls, value: list[ColumnRef]) -> list[ColumnRef]:
        if len(value) != 2:
            raise ValueError("join_columns must contain exactly two entries")
        return value


class SQL2GraphExtraction(BaseModel):
    ctes: list[SQL2GraphExtractionCTE] = Field(default_factory=list)
    output_columns: list[OutputColumn]
    filters: list[FilterSpec] = Field(default_factory=list)
    joins: list[JoinSpec] = Field(default_factory=list)
    group_by_columns: list[ColumnRef] = Field(default_factory=list)


class SQL2GraphExtractionCTE(BaseModel):
    alias: str
    output_columns: list[OutputColumn]
    filters: list[FilterSpec] = Field(default_factory=list)
    joins: list[JoinSpec] = Field(default_factory=list)
    group_by_columns: list[ColumnRef] = Field(default_factory=list)
    ctes: list[SQL2GraphExtractionCTE] = Field(default_factory=list)
