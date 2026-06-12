from typing import Optional, Dict, Any, List, TypedDict
from datetime import datetime
import os
import re
import json
import time
import hashlib

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END

from Classes.helper_classes import resolve_model_name, resolve_provider


class RefinementState(TypedDict):
    sql_input: str
    prompt: str
    attempt: int
    max_attempts: int
    response: str
    refined_sql: str
    success: bool
    last_error: Optional[str]
    validation_error: Optional[str]


class SQLRefiner:
    """
    Refines SQL scripts using LangChain and LangGraph.
    """

    def __init__(
            self,
            model: Optional[str] = None,
            provider: Optional[str] = None,
            hf_token: Optional[str] = None,
            temperature: float = 0.005,
            max_tokens: int = 2048,
            max_retries: int = 3,
            timeout: int = 120,
            retry_delay: int = 2
    ):
        """
        Initialize the SQL refiner.

        Args:
            model: Hugging Face model identifier (defaults to MODEL_NAME env var)
            provider: Model provider (defaults to PROVIDER env var)
            hf_token: Hugging Face API token
            temperature: Sampling temperature for LLM
            max_tokens: Maximum tokens in response
            max_retries: Maximum number of retry attempts
            timeout: Timeout for API calls
            retry_delay: Delay between retry attempts in seconds
        """
        # Store configuration parameters
        self.model = resolve_model_name(model, default="Qwen/Qwen2.5-Coder-32B-Instruct")
        self.provider = resolve_provider(provider, default="nebius")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.timeout = timeout
        self.retry_delay = retry_delay

        # Get Hugging Face token
        self.hf_token = hf_token or self._get_hf_token_from_env()
        if not self.hf_token:
            raise ValueError(
                "Hugging Face token is required. "
                "Set HF_TOKEN environment variable or pass hf_token parameter."
            )

        # Initialize LangChain Hugging Face chat model
        print(f"Initializing LangChain chat model: {model}")
        self.chat_model = self._create_chat_model()

        # Define the refinement prompt template
        self.refinement_prompt_template = self._create_prompt_template()
        self.refinement_graph = self._create_refinement_graph().compile()

        # Initialize session tracking
        self.session_start = datetime.now()
        self.session_stats = {
            'total_requests': 0,
            'successful_refinements': 0,
            'failed_refinements': 0,
            'total_tokens_estimated': 0,
            'total_processing_time': 0.0
        }

        # Cache for storing previous refinements
        self.refinement_cache = {}

        print("\n" + "=" * 50)
        print("SQL Refiner Initialized:")
        print(f"  Model: {self.model}")
        print(f"  Provider: {self.provider}")
        print(f"  Temperature: {self.temperature}")
        print(f"  Max tokens: {self.max_tokens}")
        print(f"  Max retries: {self.max_retries}")
        print("=" * 50 + "\n")

    def _create_chat_model(self) -> ChatHuggingFace:
        """Create LangChain ChatHuggingFace model."""
        endpoint = HuggingFaceEndpoint(
            repo_id=self.model,
            task="text-generation",
            provider=self.provider,
            huggingfacehub_api_token=self.hf_token,
            timeout=self.timeout,
            max_new_tokens=self.max_tokens,
            do_sample=self.temperature > 0,
            temperature=self.temperature,
        )
        return ChatHuggingFace(llm=endpoint)

    def _create_refinement_graph(self) -> StateGraph:
        """Create LangGraph workflow for generation + validation retries."""
        workflow = StateGraph(RefinementState)

        def generate(state: RefinementState) -> RefinementState:
            try:
                response = self.chat_model.invoke([HumanMessage(content=state["prompt"])])
                state["response"] = response.content if hasattr(response, "content") else str(response)
                state["last_error"] = None
            except Exception as exc:  # pragma: no cover - external API failures are runtime-only
                state["response"] = ""
                state["last_error"] = str(exc)
            return state

        def extract(state: RefinementState) -> RefinementState:
            if state["response"]:
                state["refined_sql"] = self._extract_sql_from_response(state["response"])
            else:
                state["refined_sql"] = ""
            return state

        def validate(state: RefinementState) -> RefinementState:
            if state.get("last_error"):
                state["success"] = False
                state["validation_error"] = state["last_error"]
                return state

            validation = self._validate_refinement(state["sql_input"], state["refined_sql"])
            state["success"] = validation.get("passed", False)
            state["validation_error"] = validation.get("error")
            return state

        def backoff(state: RefinementState) -> RefinementState:
            state["attempt"] += 1
            delay = self.retry_delay * (2 ** max(0, state["attempt"] - 1))
            print(f"Retrying in {delay}s...")
            time.sleep(delay)
            return state

        def should_retry(state: RefinementState) -> str:
            if state.get("success", False):
                return "end"
            if state["attempt"] >= state["max_attempts"]:
                return "end"
            return "retry"

        workflow.add_node("generate", generate)
        workflow.add_node("extract", extract)
        workflow.add_node("validate", validate)
        workflow.add_node("backoff", backoff)

        workflow.set_entry_point("generate")
        workflow.add_edge("generate", "extract")
        workflow.add_edge("extract", "validate")
        workflow.add_conditional_edges(
            "validate",
            should_retry,
            {"retry": "backoff", "end": END},
        )
        workflow.add_edge("backoff", "generate")

        return workflow

    def _get_hf_token_from_env(self) -> Optional[str]:
        """Get Hugging Face token from environment variables."""
        return os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACEHUB_API_TOKEN")

    def _create_prompt_template(self) -> str:
        """Create the ReAct-style prompt template."""
        return """You are an expert SQL code refiner. Refine the following SQL code by:
1. Removing redundant type casts (e.g., '1'::text -> '1')
2. Improving readability with proper indentation (4 spaces per level)
3. Simplifying complex parentheses
4. Preserving ALL business logic and structure

CRITICAL RULES:
- DO NOT change UNION/UNION ALL structure
- DO NOT alter JOIN conditions or WHERE clauses
- PRESERVE all CASE statement logic
- MAINTAIN all column mappings and data types
- KEEP all NULL handling (COALESCE, IS NULL, etc.)

EXAMPLE REFINEMENT:
Before: WHEN ((agr.agr_cred_type_cd = '1'::text) AND (agr.eks_transhes_issue_dt IS NOT NULL))
After:  WHEN agr.agr_cred_type_cd = '1' AND agr.eks_transhes_issue_dt IS NOT NULL

Output ONLY the refined SQL code with no additional text or explanations.
Do not wrap the output in markdown code blocks.

SQL TO REFINE:
{sql_input}"""

    def _generate_cache_key(self, sql_input: str) -> str:
        """Generate a cache key for SQL input."""
        content = f"{sql_input}_{self.model}_{self.temperature}"
        return hashlib.md5(content.encode()).hexdigest()

    def _analyze_sql(self, sql_input: str) -> Dict[str, Any]:
        """
        Analyze SQL code to identify common issues.

        Returns:
            Dictionary with analysis results
        """
        newline_char = '\n'
        analysis = {
            'input_hash': self._generate_cache_key(sql_input),
            'length_chars': len(sql_input),
            'length_lines': sql_input.count(newline_char) + 1,
            'issues': [],
            'complexity_metrics': {}
        }

        metrics = {
            'subqueries': max(0, sql_input.upper().count('SELECT ') - 1),
            'joins': sql_input.upper().count('JOIN '),
            'unions': sql_input.upper().count('UNION'),
            'case_statements': sql_input.upper().count('CASE'),
            'window_functions': sql_input.upper().count('OVER('),
            'ctes': sql_input.upper().count('WITH '),
        }
        analysis['complexity_metrics'] = metrics

        issues_to_check = [
            (r"'[^']+'::(?:text|bigint|integer|numeric|smallint|date)", 'Redundant type cast'),
            (r'\(\(\(\([^)]+\)\)\)\)', 'Excessive parentheses'),
            (r'\(1 = 1\)', 'Tautological condition'),
            (r'IS NOT NULL\)\)\)', 'Nested null check'),
            (r'(\$\$+|\^+|\*{2,})', 'Odd symbols in identifiers'),
            (r'\(\([^)]+\) AND \([^)]+\)\)', 'Complex AND condition'),
        ]

        for pattern, description in issues_to_check:
            matches = re.findall(pattern, sql_input)
            if matches:
                analysis['issues'].append({
                    'type': description,
                    'count': len(matches),
                    'examples': matches[:2]
                })

        return analysis

    def _build_react_prompt(self, sql_input: str, analysis: Dict[str, Any]) -> str:
        """
        Build a ReAct-style prompt for SQL refinement.

        Args:
            sql_input: Original SQL code
            analysis: Pre-analysis results

        Returns:
            Formatted prompt
        """
        max_sql_length = self.max_tokens * 2
        if len(sql_input) > max_sql_length:
            sql_input = sql_input[:max_sql_length] + "\n-- [TRUNCATED DUE TO LENGTH]"

        issues_summary = ""
        if analysis['issues']:
            issues_list = [f"- {issue['type']} (count: {issue['count']})"
                           for issue in analysis['issues']]
            issues_summary = "\nIssues identified:\n" + "\n".join(issues_list)

        subqueries = analysis['complexity_metrics'].get('subqueries', 0)
        joins = analysis['complexity_metrics'].get('joins', 0)
        case_statements = analysis['complexity_metrics'].get('case_statements', 0)
        unions = analysis['complexity_metrics'].get('unions', 0)

        prompt = f"""{self.refinement_prompt_template}

Additional context:
{issues_summary}

Complexity metrics:
- Subqueries: {subqueries}
- JOINs: {joins}
- CASE statements: {case_statements}
- UNION operations: {unions}

IMPORTANT: Keep the output SQL functionally identical to the input.

SQL TO REFINE:
{sql_input}"""

        return prompt

    def _extract_sql_from_response(self, response: str) -> str:
        """
        Extract SQL code from LLM response.

        Args:
            response: LLM response text

        Returns:
            Cleaned SQL code
        """
        sql_marker = '```sql'
        if sql_marker in response:
            pattern = r'```sql\s*\n(.*?)\n```'
            match = re.search(pattern, response, re.DOTALL)
            if match:
                return match.group(1).strip()

        code_marker = '```'
        if code_marker in response:
            pattern = r'```\s*\n(.*?)\n```'
            match = re.search(pattern, response, re.DOTALL)
            if match:
                return match.group(1).strip()

        lines = response.strip().split('\n')
        cleaned_lines = []

        skip_prefixes = ('here is', 'the refined', 'refined sql',
                         'i have', 'following is', '```', 'refined:', 'output:')

        for line in lines:
            lower_line = line.lower().strip()
            if any(lower_line.startswith(prefix) for prefix in skip_prefixes):
                continue
            cleaned_lines.append(line)

        result = '\n'.join(cleaned_lines).strip()

        if result and not result.endswith(';'):
            result += ';'

        return result

    def _validate_refinement(self, original: str, refined: str) -> Dict[str, Any]:
        """
        Validate that refinement preserves critical elements.

        Args:
            original: Original SQL code
            refined: Refined SQL code

        Returns:
            Validation results
        """
        validation = {
            'passed': True,
            'critical_checks': {},
            'warnings': [],
            'error': None
        }

        critical_elements = {
            'UNION ALL': 'UNION ALL operations',
            'SELECT': 'SELECT statements',
            'INSERT INTO': 'INSERT operations',
            'FROM': 'FROM clauses',
        }

        for element, description in critical_elements.items():
            orig_count = original.upper().count(element.upper())
            ref_count = refined.upper().count(element.upper())

            validation['critical_checks'][element] = {
                'original': orig_count,
                'refined': ref_count,
                'match': orig_count == ref_count
            }

            if not validation['critical_checks'][element]['match']:
                validation['passed'] = False
                validation['error'] = f"{description} count changed: {orig_count} -> {ref_count}"
                break

        if validation['passed']:
            warning_checks = {
                'JOIN': 'JOIN operations',
                'CASE': 'CASE statements',
                'COALESCE': 'COALESCE functions',
                'WHERE': 'WHERE clauses',
            }

            for element, description in warning_checks.items():
                orig_count = original.upper().count(element.upper())
                ref_count = refined.upper().count(element.upper())

                if orig_count != ref_count:
                    validation['warnings'].append(
                        f"{description} count changed: {orig_count} -> {ref_count}"
                    )

        return validation

    def refine_sql(
            self,
            sql_input: str,
            use_cache: bool = True,
            validate: bool = True,
            verbose: bool = False
    ) -> Dict[str, Any]:
        """
        Refine SQL code using Hugging Face LLM.

        Args:
            sql_input: SQL code to refine
            use_cache: Use caching for identical inputs
            validate: Validate the refinement
            verbose: Print progress information

        Returns:
            Dictionary with refinement results
        """
        start_time = datetime.now()
        result = {
            'success': False,
            'original_sql': sql_input,
            'refined_sql': None,
            'analysis': None,
            'validation': None,
            'stats': {},
            'error': None,
            'timestamp': start_time.isoformat(),
            'cache_hit': False,
            'method_used': 'langchain_langgraph'
        }

        try:
            # Check cache
            cache_key = self._generate_cache_key(sql_input)
            if use_cache and cache_key in self.refinement_cache:
                cached_result = self.refinement_cache[cache_key]
                if verbose:
                    cache_key_short = cache_key[:8]
                    print(f"Cache hit for input hash: {cache_key_short}...")

                result.update(cached_result)
                result['cache_hit'] = True
                result['success'] = True
                return result

            # Update session stats
            self.session_stats['total_requests'] += 1

            if verbose:
                newline_char = '\n'
                line_count = sql_input.count(newline_char) + 1
                print(f"Processing SQL ({len(sql_input)} chars, {line_count} lines)...")

            # Step 1: Analyze SQL
            result['analysis'] = self._analyze_sql(sql_input)

            # Step 2: Build and execute prompt
            prompt = self._build_react_prompt(sql_input, result['analysis'])

            if verbose:
                print(f"Running LangGraph refinement workflow on {self.model}...")
                print(f"Prompt length: {len(prompt)} chars")

            llm_start = datetime.now()
            final_state = self.refinement_graph.invoke({
                "sql_input": sql_input,
                "prompt": prompt,
                "attempt": 1,
                "max_attempts": self.max_retries,
                "response": "",
                "refined_sql": "",
                "success": False,
                "last_error": None,
                "validation_error": None,
            })
            llm_time = (datetime.now() - llm_start).total_seconds()

            response = final_state.get("response", "")
            estimated_tokens = len(response) // 4 if response else 0
            self.session_stats['total_tokens_estimated'] += estimated_tokens

            # Step 3: Extract SQL from response
            refined_sql = final_state.get("refined_sql", "")
            result['refined_sql'] = refined_sql

            # Step 4: Validate if requested
            if validate:
                result['validation'] = self._validate_refinement(sql_input, refined_sql)
                result['success'] = result['validation']['passed']

                if not result['success']:
                    result['error'] = (
                        final_state.get("validation_error")
                        or result['validation'].get('error', 'Validation failed')
                    )
            else:
                result['success'] = True

            # Step 5: Calculate statistics
            total_time = (datetime.now() - start_time).total_seconds()
            self.session_stats['total_processing_time'] += total_time

            if result['success']:
                self.session_stats['successful_refinements'] += 1

                if verbose:
                    print(f"✓ LLM refinement successful ({total_time:.1f}s)")

                    if result['refined_sql']:
                        original_len = len(sql_input)
                        refined_len = len(result['refined_sql'])
                        change = refined_len - original_len
                        if change < 0:
                            print(f"  Reduced by {abs(change)} chars ({abs(change) / original_len * 100:.1f}%)")
            else:
                self.session_stats['failed_refinements'] += 1
                if verbose:
                    print(f"✗ Refinement failed")

            result['stats'] = {
                'processing_time_seconds': round(total_time, 2),
                'llm_time_seconds': round(llm_time, 2),
                'original_length': len(sql_input),
                'refined_length': len(refined_sql),
                'compression_ratio': len(refined_sql) / len(sql_input) if len(sql_input) > 0 else 1,
                'estimated_tokens': estimated_tokens,
                'issues_found': len(result['analysis']['issues'])
            }

            # Cache the successful result
            if result['success'] and use_cache:
                cache_entry = {
                    'refined_sql': result['refined_sql'],
                    'analysis': result['analysis'],
                    'validation': result['validation'],
                    'stats': result['stats']
                }
                self.refinement_cache[cache_key] = cache_entry

        except Exception as e:
            result['error'] = str(e)
            result['success'] = False
            self.session_stats['failed_refinements'] += 1

            if verbose:
                print(f"Error during refinement: {e}")

        return result

    def batch_refine(
            self,
            sql_scripts: List[str],
            use_cache: bool = True,
            validate: bool = True,
            verbose: bool = False,
            delay_between: float = 1.0
    ) -> List[Dict[str, Any]]:
        """
        Refine multiple SQL scripts.

        Args:
            sql_scripts: List of SQL scripts
            use_cache: Use caching
            validate: Validate each refinement
            verbose: Print progress
            delay_between: Delay between requests in seconds

        Returns:
            List of refinement results
        """
        results = []

        print(f"Starting batch refinement of {len(sql_scripts)} scripts...")

        for i, sql in enumerate(sql_scripts, 1):
            if verbose:
                print(f"\n[{i}/{len(sql_scripts)}] Processing ({len(sql)} chars)...")

            result = self.refine_sql(
                sql_input=sql,
                use_cache=use_cache,
                validate=validate,
                verbose=verbose
            )
            results.append(result)

            # Add delay to avoid rate limiting
            if i < len(sql_scripts):
                time.sleep(delay_between)

        # Print summary
        success_count = sum(1 for r in results if r['success'])
        total_count = len(results)
        success_rate = (success_count / total_count * 100) if total_count > 0 else 0

        print(f"\nBatch complete. Success: {success_count}/{total_count} ({success_rate:.1f}%)")

        return results

    def get_session_stats(self) -> Dict[str, Any]:
        """Get current session statistics."""
        session_duration = (datetime.now() - self.session_start).total_seconds()

        stats = {
            'session_start': self.session_start.isoformat(),
            'session_duration_seconds': round(session_duration, 2),
            **self.session_stats,
            'cache_size': len(self.refinement_cache),
            'config': {
                'model': self.model,
                'provider': self.provider,
                'temperature': self.temperature,
                'max_tokens': self.max_tokens,
                'max_retries': self.max_retries,
                'timeout': self.timeout,
                'retry_delay': self.retry_delay
            }
        }

        # Calculate averages
        if self.session_stats['total_requests'] > 0:
            total_requests = self.session_stats['total_requests']
            total_time = self.session_stats['total_processing_time']
            successful = self.session_stats['successful_refinements']

            stats['avg_processing_time'] = round(total_time / total_requests, 2)
            stats['success_rate'] = round(successful / total_requests * 100, 1)

        return stats

    def clear_cache(self) -> None:
        """Clear the refinement cache."""
        self.refinement_cache.clear()
        print("Refinement cache cleared.")

    def save_report(self, result: Dict[str, Any], output_path: str) -> None:
        """
        Save refinement report to JSON file.

        Args:
            result: Refinement result
            output_path: Path to save report
        """
        try:
            # Create truncated samples for the report
            original_length = len(result['original_sql'])
            refined_length = len(result['refined_sql']) if result['refined_sql'] else 0

            original_sample = result['original_sql'][:200]
            if original_length > 200:
                original_sample += '...'

            refined_sample = ''
            if result['refined_sql']:
                refined_sample = result['refined_sql'][:200]
                if refined_length > 200:
                    refined_sample += '...'

            report = {
                'metadata': {
                    'timestamp': result.get('timestamp'),
                    'success': result['success'],
                    'cache_hit': result.get('cache_hit', False),
                    'model': self.model,
                    'provider': self.provider
                },
                'analysis': result.get('analysis'),
                'validation': result.get('validation'),
                'stats': result.get('stats'),
                'sql_snippets': {
                    'original_length': original_length,
                    'refined_length': refined_length,
                    'original_sample': original_sample,
                    'refined_sample': refined_sample if result['refined_sql'] else None
                }
            }

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)

            print(f"Report saved to {output_path}")

        except Exception as e:
            print(f"Failed to save report: {e}")

