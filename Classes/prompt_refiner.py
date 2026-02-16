"""
Prompt Optimisation Agent for SQL Lineage Extraction
Uses LangChain Hugging Face Chat Model (same as SQLLineageExtractor)
"""

import json
import re
import asyncio
import time
from datetime import datetime
from typing import Dict, Any, Optional, List, TypedDict, Union

from langgraph.graph import StateGraph, END
from langchain.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

# Import your existing core classes
from Classes.model_classes import SQLLineageExtractor, SQLLineageResult
from Classes.validation_classes import SQLLineageValidator


# ----------------------------------------------------------------------
# State definition (unchanged)
# ----------------------------------------------------------------------
class AgentState(TypedDict):
    sql: str
    current_prompt: str
    validation_result: Dict[str, Any]
    iteration: int
    max_iterations: int
    refined_prompts: List[str]
    validation_history: List[Dict[str, Any]]
    expected_result: Optional[Dict[str, Any]]
    should_continue: bool


# ----------------------------------------------------------------------
# Main Agent – LangChain Hugging Face version
# ----------------------------------------------------------------------
class HuggingFaceSQLLineageAgent:
    """
    Prompt optimization agent using LangChain Hugging Face chat models.
    Mirrors the exact LLM initialisation and prompt handling of SQLLineageExtractor.
    """

    def __init__(
            self,
            model: str = "Qwen/Qwen3-Coder-30B-A3B-Instruct",
            provider: str = "scaleway",
            hf_token: Optional[str] = None,
            temperature: float = 0.01,
            max_tokens: int = 2048,
            max_iterations: int = 5,
            base_extraction_template: Optional[str] = None,
            extractor: Optional[SQLLineageExtractor] = None
    ):
        """
        Args:
            model: Hugging Face model ID
            provider: Inference provider (scaleway, nebius, huggingface)
            hf_token: Hugging Face API token (required)
            temperature: Sampling temperature for reflection
            max_tokens: Max new tokens for reflection responses
            max_iterations: Maximum reflexion iterations (≤5)
            base_extraction_template: Initial prompt template for extraction.
                                       If not given, uses extractor's current template.
            extractor: Optional SQLLineageExtractor instance.
                       If not given, one is created with the same model/provider.
        """
        if not hf_token:
            raise ValueError("HF_TOKEN is required. Set it as environment variable or pass explicitly.")

        self.model = model
        self.provider = provider
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_iterations = min(max_iterations, 5)  # hard limit

        # ------------------------------------------------------------------
        # 1. LangChain Hugging Face Chat Model for REFLECTION
        # ------------------------------------------------------------------
        self.llm_endpoint = HuggingFaceEndpoint(
            repo_id=model,
            task="text-generation",
            max_new_tokens=max_tokens,
            do_sample=(temperature > 0),
            temperature=temperature,
            provider=provider,
            huggingfacehub_api_token=hf_token
        )
        self.chat_model = ChatHuggingFace(llm=self.llm_endpoint)

        # ------------------------------------------------------------------
        # 2. EXTRACTOR management – use the same pattern as SQLLineageExtractor
        # ------------------------------------------------------------------
        self.extractor = extractor
        if self.extractor is None:
            # Create a default extractor with the same parameters
            self.extractor = SQLLineageExtractor(
                model=model,
                provider=provider,
                hf_token=hf_token,
                max_new_tokens=max_tokens,
                do_sample=False,          # extraction should be deterministic
                max_retries=3,
                use_pydantic_parser=True
            )
            # Use its built‑in human prompt template as the base
            self.base_extraction_template = self.extractor.human_prompt_template
        else:
            # Use the extractor's current human prompt as base
            self.base_extraction_template = extractor.human_prompt_template

        # Override base template if explicitly given
        if base_extraction_template:
            self.base_extraction_template = base_extraction_template
            # Also update the extractor immediately to stay consistent
            if self.extractor:
                self.extractor.human_prompt_template = self._ensure_query_placeholder(base_extraction_template)
                self.extractor.chain = self.extractor._create_chain()

        # ------------------------------------------------------------------
        # 3. Reflection prompt template
        # ------------------------------------------------------------------
        self.reflection_template = PromptTemplate(
            input_variables=["current_prompt", "sql", "errors", "current_result", "iteration"],
            template="""# Task: Improve SQL Lineage Extraction Prompt

## Context
You are refining a prompt that extracts source-to-target lineage from SQL DDL statements.
The goal is to get perfect JSON output: {{"target": "schema.table", "sources": ["schema1.table1", ...]}}

## Current Prompt
{current_prompt}

## SQL Query Being Analyzed
{sql}

## Validation Issues (Iteration {iteration})
{errors}

## Current Extraction Result
{current_result}

## Improvement Instructions
Analyze the errors above and create an improved prompt that specifically addresses them.
Focus on:
1. Clarifying JSON output format requirements
2. Ensuring fully qualified names (schema.table)
3. Excluding derived tables, CTEs, subqueries, aliases
4. Removing duplicates from sources
5. Clear identification of target table

Make the prompt more precise and explicit about the rules.

## Improved Prompt (write ONLY the new prompt):
"""
        )

    # ----------------------------------------------------------------------
    # Helper: ensure the prompt contains the {query} placeholder
    # ----------------------------------------------------------------------
    @staticmethod
    def _ensure_query_placeholder(prompt: str) -> str:
        """Make sure the prompt includes the {query} placeholder required by the extractor."""
        if "{query}" not in prompt:
            prompt += "\n\nSQL Query:\n```sql\n{query}\n```\n\nYour Response (JSON ONLY):"
        return prompt

    # ----------------------------------------------------------------------
    # Helper: invoke chat model for a single string prompt
    # ----------------------------------------------------------------------
    def _generate_text(self, prompt: str) -> str:
        """Send a plain text prompt to the chat model and return the content."""
        response = self.chat_model.invoke([HumanMessage(content=prompt)])
        return response.content

    # ----------------------------------------------------------------------
    # Extraction with the internal / provided extractor
    # ----------------------------------------------------------------------
    def _safe_extract_with_retry(self, prompt: str, sql: str, max_retries: int = 3) -> Dict[str, Any]:
        """Run extraction using the agent's extractor, with retry logic."""
        extraction_start = datetime.now()

        for attempt in range(max_retries):
            try:
                print(f"\n{'=' * 40}")
                print(f"EXTRACTION ATTEMPT {attempt + 1}/{max_retries}")
                print(f"{'=' * 40}")

                # Ensure the extractor uses the current prompt (with placeholder)
                prompt_with_placeholder = self._ensure_query_placeholder(prompt)
                if hasattr(self.extractor, 'human_prompt_template'):
                    self.extractor.human_prompt_template = prompt_with_placeholder
                    self.extractor.chain = self.extractor._create_chain()

                # Run extraction
                result_dict = self.extractor.extract(sql)

                # Validate and normalise
                if "error" in result_dict:
                    raise ValueError(f"Extractor returned error: {result_dict['error']}")

                # Ensure proper structure
                if not isinstance(result_dict, dict):
                    result_dict = {"target": "", "sources": []}
                if "target" not in result_dict:
                    result_dict["target"] = ""
                if "sources" not in result_dict:
                    result_dict["sources"] = []
                if isinstance(result_dict["sources"], str):
                    result_dict["sources"] = [result_dict["sources"]]
                elif not isinstance(result_dict["sources"], list):
                    result_dict["sources"] = []

                # Clean up
                result_dict["target"] = str(result_dict.get("target", "")).strip().lower()
                result_dict["sources"] = [
                    s.strip().lower() for s in result_dict["sources"]
                    if s and str(s).strip()
                ]

                print(f"✓ Extraction successful in {(datetime.now() - extraction_start).total_seconds():.2f}s")
                return result_dict

            except Exception as e:
                print(f"✗ Extraction attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    wait = 2 * (attempt + 1)
                    print(f"Retrying in {wait}s...")
                    time.sleep(wait)
                else:
                    return {
                        "error": f"Extraction failed after {max_retries} attempts: {e}",
                        "target": "",
                        "sources": []
                    }
        return {"error": "Extraction failed", "target": "", "sources": []}

    # ----------------------------------------------------------------------
    # Public extraction interface
    # ----------------------------------------------------------------------
    def run_extraction(self, prompt: str, sql: str) -> Dict[str, Any]:
        """Run lineage extraction with the given prompt using the agent's extractor."""
        return self._safe_extract_with_retry(prompt, sql, max_retries=3)

    # ----------------------------------------------------------------------
    # Validation (unchanged logic, uses the new extraction method)
    # ----------------------------------------------------------------------
    def validate_extraction(self, prompt: str, sql: str, expected: Optional[Dict] = None) -> Dict[str, Any]:
        """Run extraction and validation with given prompt."""
        print(f"\n{'=' * 60}")
        print(f"VALIDATION")
        print(f"{'=' * 60}")

        # Run extraction
        extraction_result = self.run_extraction(prompt, sql)

        # Create a temporary wrapper for the validator
        class TempExtractor:
            def __init__(self, agent, prompt):
                self.agent = agent
                self.prompt = prompt

            def extract(self, sql_query):
                return self.agent.run_extraction(self.prompt, sql_query)

        extractor_wrapper = TempExtractor(self, prompt)

        # Run validation
        try:
            validation_result = SQLLineageValidator.run_comprehensive_validation(
                extractor=extractor_wrapper,
                sql_query=sql,
                expected_result=expected
            )
        except Exception as e:
            validation_result = {
                "status": "FAILED",
                "validation_type": "system",
                "message": f"Validation error: {e}",
                "result": extraction_result
            }

        validation_result["result"] = extraction_result
        return validation_result

    # ----------------------------------------------------------------------
    # Error formatting (unchanged)
    # ----------------------------------------------------------------------
    def _format_errors_for_reflection(self, validation_result: Dict) -> str:
        """Format validation errors for the reflection prompt."""
        if validation_result["status"] == "SUCCESS":
            if "metrics" in validation_result:
                f1 = validation_result["metrics"].get("f1_score", 0)
                if f1 >= 1.0:
                    return (f"🎉 PERFECT F1 SCORE = 1.0!\n"
                            f"Precision: {validation_result['metrics']['precision']:.4f}, "
                            f"Recall: {validation_result['metrics']['recall']:.4f}")
                else:
                    return (f"✅ SUCCESS with F1 score: {f1:.4f}\n"
                            f"Precision: {validation_result['metrics']['precision']:.4f}, "
                            f"Recall: {validation_result['metrics']['recall']:.4f}")
            return "✅ All validations passed"

        errors = []
        if validation_result.get("message"):
            errors.append(f"❌ {validation_result.get('validation_type', 'validation').upper()}: "
                         f"{validation_result['message']}")
        if "result" in validation_result and isinstance(validation_result["result"], dict):
            if "error" in validation_result["result"]:
                errors.append(f"⚠️ Extraction error: {validation_result['result']['error']}")
            else:
                current = validation_result["result"]
                errors.append(f"📊 Current extraction: target='{current.get('target', 'N/A')}', "
                             f"sources={current.get('sources', [])}")
        return "\n".join(errors) if errors else "Unknown validation error"

    # ----------------------------------------------------------------------
    # Reflection (uses LangChain chat model, not custom wrapper)
    # ----------------------------------------------------------------------
    def reflect_and_improve(self, state: Dict) -> Dict:
        """Reflect on validation results and generate improved prompt using the chat model."""
        print(f"\n{'=' * 60}")
        print(f"REFLECTION - Iteration {state['iteration'] + 1}")
        print(f"{'=' * 60}")

        if state["iteration"] >= state["max_iterations"]:
            print(f"Max iterations ({state['max_iterations']}) reached. Stopping.")
            state["should_continue"] = False
            return state

        # Check for perfect F1 score
        if (state["validation_result"]["status"] == "SUCCESS" and
                "metrics" in state["validation_result"] and
                state["validation_result"]["metrics"].get("f1_score", 0) >= 1.0):
            print(f"🎉 Perfect F1 score achieved! Stopping reflection.")
            state["should_continue"] = False
            return state

        errors = self._format_errors_for_reflection(state["validation_result"])
        current_result = json.dumps(state["validation_result"].get("result", {}), indent=2)

        reflection_prompt = self.reflection_template.format(
            current_prompt=state["current_prompt"],
            sql=state["sql"],
            errors=errors,
            current_result=current_result,
            iteration=state["iteration"] + 1
        )

        try:
            # Use the LangChain chat model to generate the improved prompt
            improved_prompt = self._generate_text(reflection_prompt).strip()

            # Clean common meta‑lines (e.g. "Improved Prompt:" or comments)
            lines = [
                line for line in improved_prompt.split('\n')
                if not line.startswith('#')
                and not line.lower().startswith('improved prompt:')
            ]
            improved_prompt = '\n'.join(lines).strip()

            if len(improved_prompt) < 50:
                raise ValueError("Generated prompt too short")

            print(f"✓ Improved prompt generated ({len(improved_prompt)} chars)")
        except Exception as e:
            print(f"✗ Reflection failed: {e}. Using fallback prompt.")
            # Fallback: add error‑specific instruction
            improved_prompt = state["current_prompt"] + (
                "\n\nIMPORTANT: Avoid the validation errors shown above. "
                "Return valid JSON with 'target' (string) and 'sources' (list of strings)."
            )

        state["current_prompt"] = improved_prompt
        state["refined_prompts"].append(improved_prompt)
        state["iteration"] += 1

        return state

    # ----------------------------------------------------------------------
    # Workflow (unchanged structure)
    # ----------------------------------------------------------------------
    def create_workflow(self) -> StateGraph:
        workflow = StateGraph(AgentState)

        def validate_node(state: AgentState) -> AgentState:
            print(f"\n{'#' * 80}")
            print(f"VALIDATE NODE - Iteration {state['iteration'] + 1}/{state['max_iterations']}")
            print(f"{'#' * 80}")

            validation_result = self.validate_extraction(
                prompt=state["current_prompt"],
                sql=state["sql"],
                expected=state.get("expected_result")
            )

            state["validation_result"] = validation_result
            state["validation_history"].append(validation_result)

            # Decision logic
            if validation_result["status"] == "FAILED":
                state["should_continue"] = True
            elif state.get("expected_result") and "metrics" in validation_result:
                f1 = validation_result["metrics"].get("f1_score", 0)
                state["should_continue"] = f1 < 1.0
            elif state.get("expected_result"):
                state["should_continue"] = True
            else:
                state["should_continue"] = (state["iteration"] < state["max_iterations"] - 1)

            if state["iteration"] >= state["max_iterations"]:
                state["should_continue"] = False

            return state

        def should_continue(state: AgentState) -> str:
            return "continue" if state.get("should_continue", True) else "end"

        workflow.add_node("validate", validate_node)
        workflow.add_node("reflect", self.reflect_and_improve)

        workflow.set_entry_point("validate")
        workflow.add_conditional_edges(
            "validate",
            should_continue,
            {"continue": "reflect", "end": END}
        )
        workflow.add_edge("reflect", "validate")

        return workflow

    # ----------------------------------------------------------------------
    # Synchronous optimisation
    # ----------------------------------------------------------------------
    def optimize_prompt_sync(
            self,
            sql: str,
            initial_prompt: Optional[str] = None,
            expected_result: Optional[Dict] = None,
            output_file: Optional[str] = None,
            verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Run the full optimisation loop and return the best prompt.

        Args:
            sql: SQL query to analyse.
            initial_prompt: Starting prompt (if None, uses base_extraction_template).
            expected_result: Ground truth lineage (optional).
            output_file: If provided, saves full iteration history to this JSON file.
            verbose: If True, prints progress after each iteration.

        Returns:
            Dictionary with optimisation results.
        """
        print(f"\n{'=' * 80}")
        print("STARTING PROMPT OPTIMIZATION (LangChain Hugging Face Agent)")
        print(f"{'=' * 80}")

        effective_initial = initial_prompt or self.base_extraction_template
        if verbose:
            print(f"\n📝 Initial prompt preview:\n{effective_initial[:200]}...\n")

        initial_state = {
            "sql": sql,
            "current_prompt": effective_initial,
            "validation_result": {},
            "iteration": 0,
            "max_iterations": self.max_iterations,
            "refined_prompts": [],
            "validation_history": [],
            "expected_result": expected_result,
            "should_continue": True
        }

        # Initialize extractor with the initial prompt
        if hasattr(self.extractor, 'human_prompt_template'):
            prompt_with_placeholder = self._ensure_query_placeholder(initial_state["current_prompt"])
            self.extractor.human_prompt_template = prompt_with_placeholder
            self.extractor.chain = self.extractor._create_chain()

        graph = self.create_workflow()
        app = graph.compile()

        final_state = initial_state
        for i in range(self.max_iterations):
            print(f"\n{'#' * 80}")
            print(f"ITERATION {i + 1} / {self.max_iterations}")
            print(f"{'#' * 80}")

            final_state = app.invoke(final_state)

            if verbose and final_state.get("validation_result"):
                val = final_state["validation_result"]
                f1 = val.get("metrics", {}).get("f1_score", 0) if "metrics" in val else 0
                target = val.get("result", {}).get("target", "N/A")
                src_count = len(val.get("result", {}).get("sources", []))
                print(f"\n📊 Iteration {i + 1} summary:")
                print(f"   F1 score   : {f1:.4f}")
                print(f"   Target      : {target}")
                print(f"   Sources     : {src_count}")
                print(f"   Prompt preview (first 200 chars):\n{final_state['current_prompt'][:200]}...\n")

            if not final_state.get("should_continue", True):
                print(f"\n⏹️ Stopping early at iteration {i + 1} – should_continue = False")
                break

        # Determine the best prompt (F1=1 or highest F1)
        perfect_prompt = None
        best_f1 = -1.0
        best_prompt = initial_state["current_prompt"]
        best_iter = -1

        for i, val in enumerate(final_state.get("validation_history", [])):
            if val["status"] == "SUCCESS" and "metrics" in val:
                f1 = val["metrics"].get("f1_score", 0)
                if f1 > best_f1:
                    best_f1 = f1
                    best_iter = i
                    if i == 0:
                        best_prompt = initial_state["current_prompt"]
                    else:
                        best_prompt = final_state["refined_prompts"][i - 1]
                if f1 >= 1.0:
                    perfect_prompt = best_prompt
                    best_f1 = f1
                    break

        if perfect_prompt is None:
            perfect_prompt = best_prompt

        result = {
            "optimized_prompt": perfect_prompt,
            "initial_prompt": effective_initial,
            "f1_score": best_f1,
            "is_perfect": best_f1 >= 1.0,
            "iterations": final_state["iteration"],
            "iteration_found": best_iter + 1,
            "final_validation": final_state.get("validation_result", {}),
            "all_prompts": final_state["refined_prompts"],
            "validation_history": final_state["validation_history"],
            "extractor_used": True
        }

        if output_file:
            self._save_history_to_file(result, output_file, sql, expected_result)

        return result

    async def optimize_prompt(self, *args, **kwargs) -> Dict[str, Any]:
        """Async wrapper (unchanged)."""
        return await asyncio.get_event_loop().run_in_executor(
            None, lambda: self.optimize_prompt_sync(*args, **kwargs)
        )

    def _save_history_to_file(self, result: Dict, output_file: str, sql: str, expected: Optional[Dict]):
        """Save the full optimisation history to a JSON file."""
        import json
        from datetime import datetime

        history = {
            "timestamp": datetime.now().isoformat(),
            "sql": sql,
            "expected_result": expected,
            "optimization_result": result,
            "iteration_details": []
        }

        # Iteration 0 uses the initial prompt (not base_extraction_template)
        initial_prompt = result.get("initial_prompt", self.base_extraction_template)

        # Add per-iteration details
        for i, val in enumerate(result["validation_history"]):
            if i == 0:
                prompt = initial_prompt
            else:
                # refined prompts are stored for iterations 1..n
                prompt = result["all_prompts"][i - 1] if i - 1 < len(result["all_prompts"]) else ""

            history["iteration_details"].append({
                "iteration": i + 1,
                "prompt": prompt,
                "validation": val
            })

        with open(output_file, "w") as f:
            json.dump(history, f, indent=2)

        print(f"\n💾 Optimisation history saved to {output_file}")

# ----------------------------------------------------------------------
# AdvancedReflexionAgent – same pattern, extended with few‑shot examples
# ----------------------------------------------------------------------
class AdvancedReflexionAgent(HuggingFaceSQLLineageAgent):
    """Extended version with few‑shot examples and enhanced reflection prompt."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.few_shot_examples = [
            {
                "problem": "Extracting CTE as source",
                "bad_prompt": "Extract all tables mentioned in the SQL",
                "good_prompt": "Extract only base tables from FROM and JOIN clauses. Exclude CTEs defined in WITH clauses."
            },
            {
                "problem": "Not fully qualified names",
                "bad_prompt": "List the source tables",
                "good_prompt": "Extract fully qualified table names in format 'schema.table'. If schema is not specified, infer from context or use 'public' as default."
            },
            {
                "problem": "Including derived tables",
                "bad_prompt": "Get all tables used",
                "good_prompt": "Identify only base physical tables. Exclude derived tables, subqueries, aliases like t1, t2, subq, etc."
            }
        ]

        self.enhanced_reflection_template = PromptTemplate(
            input_variables=["current_prompt", "sql", "errors", "current_result", "iteration", "examples"],
            template="""# PROMPT OPTIMIZATION FOR SQL LINEAGE EXTRACTION

## Current Prompt (Iteration {iteration})
{current_prompt}

## SQL Query
{sql}

## Validation Issues
{errors}

## Current Extraction
{current_result}

## Example Improvements
{examples}

## Instructions
Analyze the specific validation errors above. Compare with the examples to understand common issues.
Create a new prompt that:
1. Explicitly addresses the specific errors shown above
2. Is more precise and unambiguous than the current prompt
3. Includes clear formatting instructions for JSON output
4. Specifies rules to avoid the validation failures

Write ONLY the improved prompt, no explanations:"""
        )

    def reflect_and_improve(self, state: Dict) -> Dict:
        """Reflection with few‑shot examples, using the same chat model."""
        print(f"\n{'=' * 60}")
        print(f"ADVANCED REFLECTION - Iteration {state['iteration'] + 1}")
        print(f"{'=' * 60}")

        if state["iteration"] >= state["max_iterations"]:
            state["should_continue"] = False
            return state

        if (state["validation_result"]["status"] == "SUCCESS" and
                "metrics" in state["validation_result"] and
                state["validation_result"]["metrics"].get("f1_score", 0) >= 1.0):
            state["should_continue"] = False
            return state

        errors = self._format_errors_for_reflection(state["validation_result"])
        current_result = json.dumps(state["validation_result"].get("result", {}), indent=2)

        examples_text = "\n".join([
            f"Problem: {ex['problem']}\nBad: {ex['bad_prompt']}\nGood: {ex['good_prompt']}\n"
            for ex in self.few_shot_examples
        ])

        reflection_prompt = self.enhanced_reflection_template.format(
            current_prompt=state["current_prompt"],
            sql=state["sql"],
            errors=errors,
            current_result=current_result,
            iteration=state["iteration"] + 1,
            examples=examples_text
        )

        try:
            improved_prompt = self._generate_text(reflection_prompt).strip()
            # Clean meta‑lines
            lines = [
                line for line in improved_prompt.split('\n')
                if not line.startswith('#') and not line.lower().startswith('improved prompt:')
            ]
            improved_prompt = '\n'.join(lines).strip()
        except Exception as e:
            print(f"✗ Advanced reflection failed: {e}. Using fallback.")
            improved_prompt = state["current_prompt"] + (
                "\n\nADDITIONAL RULE: Avoid the validation errors shown above."
            )

        state["current_prompt"] = improved_prompt
        state["refined_prompts"].append(improved_prompt)
        state["iteration"] += 1

        return state