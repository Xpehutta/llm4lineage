"""Doc agent: turn documentation / column metadata into structured labels."""

from __future__ import annotations

import json
from typing import Any

from pydantic import ValidationError

from Classes.agents._json_util import extract_json
from Classes.agents.models import ColumnDocLabel, DocLabels
from Classes.pipeline.core.llm_interface import LLMInterface

__all__ = ["DocAgent"]

_SYSTEM = (
    "You are a data documentation agent. Given documentation text and optional "
    "column metadata, return structured JSON labels. Mark is_pii=true only when "
    "the text clearly indicates personal data (names, emails, phones, IDs, etc.). "
    "Do not invent owners or PII flags. Return ONLY valid JSON:\n"
    '{"columns":[{"column":"...","is_pii":false,"owner":"...",'
    '"description":"...","sensitivity":"","tags":[]}],'
    '"owner":"","description":""}'
)


class DocAgent:
    """Extract ``is_pii`` / owner / description labels via structured JSON."""

    def __init__(self, llm: LLMInterface):
        self.llm = llm

    def label(
        self,
        documentation: str,
        *,
        column_metadata: list[dict[str, Any]] | dict[str, Any] | None = None,
    ) -> DocLabels:
        """Parse documentation into :class:`DocLabels` (Pydantic + ``json.loads``)."""
        prompt = self._build_prompt(documentation, column_metadata)
        raw = self.llm.invoke(prompt)
        return self._parse_response(raw, column_metadata)

    def apply_to_columns(
        self,
        columns: list[dict[str, Any]],
        labels: DocLabels,
    ) -> list[dict[str, Any]]:
        """Merge labels into column dicts, setting ``is_pii`` / owner / description."""
        by_name = {
            (item.column or "").lower(): item
            for item in labels.columns
            if item.column
        }
        enriched: list[dict[str, Any]] = []
        for col in columns:
            row = dict(col)
            name = str(row.get("name") or row.get("column") or "").lower()
            hit = by_name.get(name)
            if hit is None and len(labels.columns) == 1 and not name:
                hit = labels.columns[0]
            if hit is not None:
                row["is_pii"] = hit.is_pii
                if hit.owner:
                    row["owner"] = hit.owner
                elif labels.owner:
                    row["owner"] = labels.owner
                if hit.description:
                    row["description"] = hit.description
                elif labels.description and "description" not in row:
                    row["description"] = labels.description
            else:
                row.setdefault("is_pii", False)
                if labels.owner and "owner" not in row:
                    row["owner"] = labels.owner
            enriched.append(row)
        return enriched

    def _build_prompt(
        self,
        documentation: str,
        column_metadata: list[dict[str, Any]] | dict[str, Any] | None,
    ) -> str:
        meta_json = json.dumps(column_metadata or [], ensure_ascii=False, indent=2)
        return (
            f"{_SYSTEM}\n\n"
            f"### Documentation\n{documentation or ''}\n\n"
            f"### Column metadata\n{meta_json}\n"
        )

    def _parse_response(
        self,
        raw: str,
        column_metadata: list[dict[str, Any]] | dict[str, Any] | None,
    ) -> DocLabels:
        try:
            data = extract_json(raw)
        except ValueError:
            return DocLabels()

        if isinstance(data, list):
            data = {"columns": data}
        if not isinstance(data, dict):
            return DocLabels()

        columns_raw = data.get("columns")
        if columns_raw is None and any(k in data for k in ("is_pii", "column", "description", "owner")):
            columns_raw = [data]
        if columns_raw is None:
            columns_raw = []

        columns: list[ColumnDocLabel] = []
        for row in columns_raw:
            label = self._coerce_column(row)
            if label is not None:
                columns.append(label)

        # If the model only returned a single label without a column name,
        # attach it to the first metadata column when available.
        if (
            len(columns) == 1
            and not columns[0].column
            and isinstance(column_metadata, list)
            and column_metadata
        ):
            first = column_metadata[0]
            name = str(first.get("name") or first.get("column") or "")
            if name:
                columns[0] = columns[0].model_copy(update={"column": name})

        return DocLabels(
            columns=columns,
            owner=str(data.get("owner") or ""),
            description=str(data.get("description") or ""),
        )

    @staticmethod
    def _coerce_column(row: Any) -> ColumnDocLabel | None:
        if not isinstance(row, dict):
            return None
        try:
            return ColumnDocLabel.model_validate(
                {
                    "column": str(row.get("column") or row.get("name") or ""),
                    "is_pii": bool(row.get("is_pii", False)),
                    "owner": str(row.get("owner") or ""),
                    "description": str(row.get("description") or ""),
                    "sensitivity": str(row.get("sensitivity") or ""),
                    "tags": list(row.get("tags") or []),
                }
            )
        except (ValidationError, TypeError, ValueError):
            return None
