import json
from typing import List, Dict, Any
from dataclasses import dataclass
from pydantic import BaseModel, Field, field_validator, validator

# Hugging Face
from huggingface_hub import InferenceClient

@dataclass
class SQLLineageResult:
    """Data class for SQL lineage extraction result"""
    target: str
    sources: List[str]

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary"""
        return {"target": self.target, "sources": self.sources}

    def to_json(self) -> str:
        """Convert result to JSON string"""
        return json.dumps(self.to_dict(), indent=2)

    @property
    def source_count(self) -> int:
        """Get number of source tables"""
        return len(self.sources)

    def add_source(self, source: str) -> None:
        """Add a source table if not already present"""
        if source and source not in self.sources:
            self.sources.append(source)

    def __str__(self) -> str:
        """String representation"""
        return f"Target: {self.target}, Sources: {', '.join(self.sources)}"


# Simple direct wrapper for Hugging Face InferenceClient
class HuggingFaceLLMWrapper:
    """Simple wrapper for Hugging Face InferenceClient to work with LangChain"""

    def __init__(
            self,
            model: str,
            hf_token: str,
            provider: str = "nebius",
            temperature: float = 0.005,
            max_tokens: int = 2048,
            timeout: int = 120
    ):
        self.model = model
        self.hf_token = hf_token
        self.provider = provider
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout

        # Initialize client
        self.client = InferenceClient(
            provider=provider,
            api_key=hf_token,
            timeout=timeout
        )

    def __call__(self, prompt: str) -> str:
        """Call the model with a prompt"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            return response.choices[0].message.content
        except Exception as e:
            raise Exception(f"Error calling model: {e}")


class SQLDependencies(BaseModel):
    """Pydantic model for SQL lineage output"""
    target: str = Field(description="The main object being created or modified (fully qualified name)")
    sources: List[str] = Field(description="List of DISTINCT base tables/views (fully qualified names)")

    @field_validator('target')
    def normalize_target(cls, v):
        """Normalize target name"""
        if v:
            # Remove quotes and normalize
            v = v.replace('"', '').replace("'", "")
        return v.lower() if v else v

    @validator('sources', each_item=True)
    def normalize_source(cls, v):
        """Normalize source names"""
        if v:
            # Remove quotes and normalize
            v = v.replace('"', '').replace("'", "")
        return v.lower() if v else v

    def to_lineage_result(self) -> 'SQLLineageResult':
        """Convert to SQLLineageResult"""
        return SQLLineageResult(target=self.target, sources=self.sources)


class HuggingFaceLLMAdapter:
    """Adapter to make HuggingFaceLLMWrapper work with reflexion agent pattern"""

    def __init__(self, wrapper: HuggingFaceLLMWrapper):
        self.wrapper = wrapper

    def invoke(self, prompt: str) -> str:
        """LangChain-style invoke method"""
        return self.wrapper(prompt)

    def __call__(self, prompt: str) -> str:
        """Make it callable like the wrapper"""
        return self.wrapper(prompt)