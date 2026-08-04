"""Parse raw SQL into a sqlglot AST."""

import logging

import sqlglot
from sqlglot import exp
from sqlglot.errors import ParseError

from Classes.pipeline.exceptions import ParsingError

logger = logging.getLogger(__name__)


class SQLParser:
    """Parse a raw SQL string into a sqlglot AST."""

    def __init__(self, dialect: str = "postgres", error_on_incomplete: bool = True):
        self.dialect = dialect
        self.error_on_incomplete = error_on_incomplete

    def parse(self, sql: str) -> exp.Expression:
        """Return the AST for *sql*.

        Raises:
            ParsingError: If sqlglot cannot parse the input.
        """
        if not sql or not sql.strip():
            raise ParsingError("Input SQL is empty or whitespace-only.")

        try:
            tree = sqlglot.parse_one(
                sql,
                dialect=self.dialect,
                error_level=sqlglot.ErrorLevel.RAISE
                if self.error_on_incomplete
                else sqlglot.ErrorLevel.WARN,
            )
        except ParseError as exc:
            logger.error("SQL parse failure: %s", exc)
            raise ParsingError(f"Failed to parse SQL: {exc}") from exc

        if tree is None:
            raise ParsingError("Parser returned None for the given SQL.")

        logger.debug("Parsed SQL into AST root: %s", type(tree).__name__)
        return tree
