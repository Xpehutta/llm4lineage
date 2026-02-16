import os
import re
import json
import time
from typing import List, Dict, Any, Optional, Union


# LangChain HuggingFace imports
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import BaseOutputParser, PydanticOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda



# Helper Classes
from Classes.helper_classes import SQLLineageResult, SQLDependencies


class SQLLineageOutputParser(BaseOutputParser[SQLDependencies]):
    """Custom output parser for SQL lineage extraction"""

    def parse(self, text: str) -> SQLDependencies:
        """Parse LLM output into structured format"""
        # Clean response content
        content = re.sub(r'[\x00-\x1F]+', ' ', text).strip()

        # Try to parse as JSON first
        json_match = re.search(r'\{[\s\S]*\}', content)
        if json_match:
            json_str = json_match.group(0)
            try:
                data = json.loads(json_str)

                # Extract target and sources
                target = data.get('target', '')
                sources = data.get('sources', [])

                # Ensure sources is a list
                if isinstance(sources, str):
                    sources = [sources]
                elif not isinstance(sources, list):
                    sources = []

                # Normalize
                if target:
                    target = target.replace('"', '').replace("'", "").lower()
                sources = [s.replace('"', '').replace("'", "").lower() for s in sources if s]

                return SQLDependencies(target=target, sources=sources)

            except json.JSONDecodeError:
                # Fall back to regex extraction
                pass

        # Fallback: Try to extract using regex patterns
        target_patterns = [
            r'"target"\s*:\s*"([^"]+)"',
            r"'target'\s*:\s*'([^']+)'",
            r'target["\']?\s*:\s*["\']?([^"\',\s}]+)',
        ]

        source_patterns = [
            r'"sources"\s*:\s*\[(.*?)\]',
            r"'sources'\s*:\s*\[(.*?)\]",
            r'sources["\']?\s*:\s*\[(.*?)\]',
        ]

        target = ''
        sources = []

        # Extract target
        for pattern in target_patterns:
            match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
            if match:
                target = match.group(1).strip().strip('"\'').lower()
                break

        # Extract sources
        for pattern in source_patterns:
            match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
            if match:
                sources_str = match.group(1)
                # Parse sources list
                sources = re.findall(r'"([^"]+)"', sources_str)
                if not sources:
                    sources = re.findall(r"'([^']+)'", sources_str)
                if not sources:
                    sources = re.findall(r'([^,\]\s]+)', sources_str)
                sources = [s.strip().strip('"\'').lower() for s in sources if s.strip()]
                break

        return SQLDependencies(target=target, sources=sources)

    @property
    def _type(self) -> str:
        return "sql_lineage_parser"


class SQLLineageExtractor:
    """SQL lineage extractor using langchain_huggingface"""

    def __init__(
            self,
            model: str = "Qwen/Qwen3-Coder-30B-A3B-Instruct",
            provider: str = "scaleway",
            hf_token: Optional[str] = None,
            max_new_tokens: int = 2048,
            do_sample: bool = False,
            max_retries: int = 5,
            use_pydantic_parser: bool = True,
            human_prompt_template: Optional[str] = None
    ):
        """
        Initialize the SQL lineage extractor with langchain_huggingface.

        Args:
            model: Hugging Face model identifier
            provider: Model provider (scaleway, nebius, huggingface, etc.)
            hf_token: Hugging Face API token (defaults to HF_TOKEN env var)
            max_new_tokens: Maximum tokens in response
            do_sample: Whether to use sampling
            max_retries: Maximum number of retry attempts
            use_pydantic_parser: Whether to use Pydantic output parser
        """
        # Get Hugging Face token from parameter or environment
        self.hf_token = hf_token or os.environ.get("HF_TOKEN")

        if not self.hf_token:
            raise ValueError(
                "Hugging Face token not found. "
                "Please provide hf_token parameter or set HF_TOKEN environment variable."
            )

        self.model = model
        self.provider = provider
        self.max_new_tokens = max_new_tokens
        self.do_sample = do_sample
        self.max_retries = max_retries
        self.use_pydantic_parser = use_pydantic_parser

        # Initialize LLM and chat model
        self.llm = self._initialize_llm()
        self.chat_model = ChatHuggingFace(llm=self.llm)

        # Create system prompt template
        self.system_prompt = """You are a SQL lineage extraction expert. Extract source-to-target lineage from SQL statements and return ONLY valid JSON."""

        # Create human prompt template
        self.human_prompt_template = self._create_human_prompt_template() if human_prompt_template is None else human_prompt_template

        # Create output parser
        self.output_parser = self._create_output_parser()

        # Create the processing chain
        self.chain = self._create_chain()

    def _initialize_llm(self) -> HuggingFaceEndpoint:
        """Initialize HuggingFaceEndpoint LLM"""
        try:
            llm = HuggingFaceEndpoint(
                repo_id=self.model,
                task="text-generation",
                max_new_tokens=self.max_new_tokens,
                do_sample=self.do_sample,
                # repetition_penalty=1.03,  # Optional
                provider=self.provider,
                huggingfacehub_api_token=self.hf_token
            )
            return llm
        except Exception as e:
            raise RuntimeError(f"Failed to initialize HuggingFaceEndpoint: {str(e)}")

    def _create_human_prompt_template(self) -> str:
        """Create the human prompt template for SQL lineage extraction"""
        return """
Please extract source-to-target lineage from the SQL INSERT statement below. Return ONLY valid JSON with the following structure:

```json
{
  "target": "schema.table",
  "sources": ["schema1.table1", "schema2.table2", ...]
}
```

1. **Target**: Identify the main table being inserted into (fully qualified name).
2. **Sources**: 
   - Include only base tables and views (no subqueries, CTEs, or derived tables).
   - Use fully qualified names (schema.table).
   - Remove any duplicate entries.
   - Ensure all source names are distinct.

```json
{
  "target": "my_schema.my_table",
  "sources": ["schema1.table1", "schema2.table2"]
}
```
"""

    def _create_output_parser(self):
        """Create appropriate output parser"""
        if self.use_pydantic_parser:
            try:
                parser = PydanticOutputParser(pydantic_object=SQLDependencies)
                return parser
            except Exception:
                # Fall back to custom parser
                return SQLLineageOutputParser()
        else:
            return SQLLineageOutputParser()

    def _create_chain(self):
        """Create LangChain processing pipeline"""

        def format_messages(sql_query: str) -> List[Union[SystemMessage, HumanMessage]]:
            """Format SQL query into chat messages"""
            # Escape curly braces for safe formatting
            escaped_query = sql_query.replace('{', '{{').replace('}', '}}')

            # Create human prompt with SQL
            human_prompt = f"{self.human_prompt_template}\n\nSQL Query:\n```sql\n{escaped_query}\n```"

            return [
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=human_prompt)
            ]

        def invoke_model(messages: List[Union[SystemMessage, HumanMessage]]) -> str:
            """Invoke the chat model with messages"""
            response = self.chat_model.invoke(messages)
            return response.content

        def parse_response(response: str) -> Dict[str, Any]:
            """Parse the model response"""
            if self.use_pydantic_parser and isinstance(self.output_parser, PydanticOutputParser):
                try:
                    result = self.output_parser.parse(response)
                    return result.dict()
                except Exception:
                    # Fallback to custom parser
                    fallback_parser = SQLLineageOutputParser()
                    result = fallback_parser.parse(response)
                    return result.dict()
            else:
                result = self.output_parser.parse(response)
                return result.dict()

        # Create the chain
        chain = (
                RunnablePassthrough()
                | RunnableLambda(format_messages)
                | RunnableLambda(invoke_model)
                | RunnableLambda(parse_response)
        )

        return chain

    def extract(self, sql_query: str) -> Dict[str, Any]:
        """
        Extract lineage from SQL query.

        Args:
            sql_query: SQL query to analyze

        Returns:
            Dictionary with 'target' and 'sources' keys
        """
        for attempt in range(1, self.max_retries + 1):
            try:
                # Execute the chain
                result = self.chain.invoke(sql_query)

                # Validate the result
                if not result.get('target'):
                    raise ValueError("No target found in extraction result")

                return result

            except Exception as e:
                wait_time = min(30, 2 ** attempt)
                print(f"Attempt {attempt}/{self.max_retries} failed: {str(e)}. Retrying in {wait_time}s...")
                time.sleep(wait_time)

        return {"error": "Max retries exceeded", "target": "", "sources": []}

    def extract_with_result(self, sql_query: str) -> SQLLineageResult:
        """
        Extract lineage and return SQLLineageResult object.

        Args:
            sql_query: SQL query to analyze

        Returns:
            SQLLineageResult object
        """
        result_dict = self.extract(sql_query)
        return SQLLineageResult(
            target=result_dict.get("target", ""),
            sources=result_dict.get("sources", [])
        )

    def batch_extract(self, sql_queries: List[str]) -> List[Dict[str, Any]]:
        """
        Extract lineage from multiple SQL queries.

        Args:
            sql_queries: List of SQL queries to analyze

        Returns:
            List of lineage dictionaries
        """
        results = []
        for query in sql_queries:
            try:
                result = self.extract(query)
                results.append(result)
            except Exception as e:
                results.append({
                    "error": str(e),
                    "target": "",
                    "sources": [],
                    "query": query[:100]  # Truncate for logging
                })
        return results

    def get_config_info(self) -> Dict[str, Any]:
        """Get information about the configuration"""
        return {
            "model": self.model,
            "provider": self.provider,
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.do_sample,
            "max_retries": self.max_retries,
            "use_pydantic_parser": self.use_pydantic_parser
        }

    def test_connection(self) -> bool:
        """Test connection to the model"""
        try:
            # Simple test query
            test_query = "SELECT * FROM test.table"
            result = self.extract(test_query)
            return "error" not in result
        except Exception:
            return False


def create_sql_lineage_extractor(
        model: str = "Qwen/Qwen3-Coder-30B-A3B-Instruct",
        provider: str = "scaleway",
        hf_token: Optional[str] = None,
        **kwargs
) -> SQLLineageExtractor:
    """
    Factory function to create a SQLLineageExtractor.

    Args:
        model: Model name
        provider: Provider name
        hf_token: Hugging Face token
        **kwargs: Additional parameters for SQLLineageExtractor

    Returns:
        SQLLineageExtractor instance
    """
    return SQLLineageExtractor(
        model=model,
        provider=provider,
        hf_token=hf_token,
        **kwargs
    )
