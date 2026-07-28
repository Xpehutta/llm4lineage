"""Shared fixtures for sql_pipeline tests."""

import pytest

from Classes.pipeline.models.config import Config

SAMPLE_SQL = (
    "SELECT u.name, SUM(o.amount) AS total "
    "FROM users u JOIN orders o ON u.id = o.user_id "
    "GROUP BY u.name"
)

SAMPLE_SQL_SIMPLE = "SELECT a, b FROM t"

SAMPLE_SQL_STAR = "SELECT * FROM users u JOIN orders o ON u.id = o.user_id"

SAMPLE_SQL_ALIAS = "SELECT x AS y FROM t"

INVALID_SQL = "THIS IS NOT SQL"


@pytest.fixture
def mock_config() -> Config:
    return Config(llm_provider="mock")


@pytest.fixture
def sample_sql() -> str:
    return SAMPLE_SQL
