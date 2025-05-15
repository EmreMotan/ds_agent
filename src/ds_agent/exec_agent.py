from typing import Any, Dict
from .tools import load_tools
from pathlib import Path
import nbformat
from . import kernel_runner
from . import notebook_editor
import os
import openai
import json
import logging
import yaml
from .episode import load_episode
import re
import keyword
import sys


class ExecAgent:
    """
    Execution Agent that plans, applies, and runs analysis episodes using the tool registry and notebook engine.
    """

    def __init__(self, model: str = "gpt-4.1", log_level: str = "INFO"):
        self.model = model
        self.tools = load_tools()
        self.openai_api_key = os.environ.get("OPENAI_API_KEY")
        if self.openai_api_key:
            openai.api_key = self.openai_api_key
        self.logger = logging.getLogger(f"ExecAgent.{id(self)}")
        handler = logging.StreamHandler()
        formatter = logging.Formatter("[%(levelname)s] %(message)s")
        handler.setFormatter(formatter)
        if not self.logger.hasHandlers():
            self.logger.addHandler(handler)
        self.logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
        # TODO: Initialize LLM client, notebook editor, kernel runner

    def _load_schema(self):
        """Load schema from schema_cache.yaml and return as a tuple: (tables, valid_sources)"""
        schema_path = Path(__file__).parent.parent.parent / "schema_cache.yaml"
        with open(schema_path, "r") as f:
            cache = yaml.safe_load(f)
        tables = {}
        valid_sources = set()
        for source, src_data in cache.get("sources", {}).items():
            valid_sources.add(source)
            for tbl_name, tbl_data in src_data.get("tables", {}).items():
                columns = tbl_data.get("columns", {})
                table_desc = tbl_data.get("table_desc", "")
                tables[tbl_name] = {"columns": columns, "table_desc": table_desc, "source": source}
        return tables, valid_sources

    def _format_schema_for_prompt(self, schema: dict, valid_sources: set) -> str:
        """Format schema as a string for LLM prompt, including table/column descriptions and value samples."""
        lines = ["Valid source names:"]
        lines.append(", ".join(sorted(valid_sources)))
        lines.append("")
        lines.append("Available tables:")
        for tbl, meta in schema.items():
            lines.append(f"- {tbl}: {meta.get('table_desc','')}")
            lines.append(f"  Source: {meta.get('source','')}")
            lines.append(f"  Columns:")
            for col, cmeta in meta["columns"].items():
                dtype = cmeta.get("dtype", "")
                cdesc = cmeta.get("col_desc", "")
                values = cmeta.get("values", None)
                val_str = f"; sample values: {values[:10]}" if values else ""
                lines.append(f"    - {col} ({dtype}): {cdesc}{val_str}")
        return "\n".join(lines)

    def _validate_plan(self, plan: dict, schema: dict, valid_sources: set, episode_id: str = None):
        """Validate that all referenced tables, columns, sources, and episode_id in the plan exist in the schema, including columns created by previous steps."""
        # Track available columns for each step by id
        step_columns = {}
        step_id_map = {step["id"]: step for step in plan.get("steps", [])}
        for step in plan.get("steps", []):
            args = step.get("args", {})
            tool = step.get("tool")
            step_id = step["id"]
            # Determine available columns for this step
            if tool == "load_table":
                table = args.get("table")
                cols = args.get("cols")
                if table and table not in schema:
                    raise ValueError(
                        f"Step {step_id}: Table '{table}' not found in schema.\nSchema: {list(schema.keys())}\nStep: {step}"
                    )
                table_columns = schema[table]["columns"].keys() if table else []
                if cols:
                    missing = [c for c in cols if c not in table_columns]
                    if missing:
                        raise ValueError(
                            f"Step {step_id}: Columns {missing} not found in table '{table}'.\nTable columns: {list(table_columns)}\nStep: {step}"
                        )
                    step_columns[step_id] = set(cols)
                else:
                    step_columns[step_id] = set(table_columns)
            elif tool == "assign_column":
                input_df = args.get("df")
                new_col = args.get("column")
                input_cols = set()
                if isinstance(input_df, str) and input_df in step_columns:
                    input_cols = set(step_columns.get(input_df, []))
                # Always add the new column, even if not present in input
                step_columns[step_id] = input_cols | {new_col}
            elif tool == "select_columns":
                input_df = args.get("df")
                columns = args.get("columns")
                input_cols = set()
                if isinstance(input_df, str) and input_df in step_columns:
                    input_cols = set(step_columns.get(input_df, []))
                if columns:
                    missing = [c for c in columns if c not in input_cols]
                    if missing:
                        raise ValueError(
                            f"Step {step_id}: Columns {missing} not found in input DataFrame.\nAvailable columns: {list(input_cols)}\nStep: {step}"
                        )
                    step_columns[step_id] = set(columns)
                else:
                    step_columns[step_id] = input_cols
            elif tool == "merge":
                # For merge, output columns are the union of columns from both input DataFrames
                df1 = args.get("df1")
                df2 = args.get("df2")
                cols1 = set()
                cols2 = set()
                if isinstance(df1, str) and df1 in step_columns:
                    cols1 = set(step_columns.get(df1, []))
                if isinstance(df2, str) and df2 in step_columns:
                    cols2 = set(step_columns.get(df2, []))
                step_columns[step_id] = cols1 | cols2
            elif tool == "concat":
                # For concat, output columns are the union of all input DataFrames
                dfs = args.get("dfs", [])
                all_cols = set()
                for df_ref in dfs:
                    if isinstance(df_ref, str) and df_ref in step_columns:
                        all_cols |= set(step_columns.get(df_ref, []))
                step_columns[step_id] = all_cols
            elif tool == "groupby_aggregate":
                input_df = args.get("df")
                groupby_cols = args.get("groupby_cols", [])
                agg_dict = args.get("agg_dict", {})
                input_cols = set()
                if isinstance(input_df, str) and input_df in step_columns:
                    input_cols = set(step_columns.get(input_df, []))
                # Output columns: groupby_cols + key_agg for each agg
                output_cols = set(groupby_cols)
                for k, v in (agg_dict or {}).items():
                    if isinstance(v, list):
                        for agg in v:
                            output_cols.add(f"{k}_{agg}")
                    else:
                        output_cols.add(f"{k}_{v}")
                step_columns[step_id] = output_cols
            else:
                input_df = args.get("df")
                input_cols = set()
                if isinstance(input_df, str) and input_df in step_columns:
                    input_cols = set(step_columns.get(input_df, []))
                step_columns[step_id] = input_cols
            # Validate groupby_aggregate, etc. for columns
            for key in ["columns", "groupby_cols", "by", "sort_col", "values", "x", "y"]:
                if key in args and isinstance(args[key], list):
                    for col in args[key]:
                        if (
                            not any(col in t["columns"].keys() for t in schema.values())
                            and col not in input_cols
                        ):
                            raise ValueError(
                                f"Step {step_id}: Column '{col}' not found in any table or previous step.\nSchema: {schema}\nAvailable columns: {list(input_cols)}\nStep: {step}"
                            )
                elif key in args and isinstance(args[key], str):
                    col = args[key]
                    if (
                        not any(col in t["columns"].keys() for t in schema.values())
                        and col not in input_cols
                    ):
                        raise ValueError(
                            f"Step {step_id}: Column '{col}' not found in any table or previous step.\nSchema: {schema}\nAvailable columns: {list(input_cols)}\nStep: {step}"
                        )
            source = args.get("source")
            if source and source not in valid_sources:
                raise ValueError(
                    f"Step {step_id}: Source '{source}' not found in valid sources.\nValid sources: {sorted(valid_sources)}\nStep: {step}"
                )
            eid = args.get("episode_id")
            if eid and episode_id and eid != episode_id:
                raise ValueError(
                    f"Step {step_id}: episode_id '{eid}' does not match current episode_id '{episode_id}'.\nStep: {step}"
                )
            # After determining step_columns[step_id]:
            self.logger.debug(
                f"Step {step_id} ({tool}): available columns: {sorted(step_columns[step_id])}"
            )
        # Validate depends_on
        for step in plan.get("steps", []):
            depends_on = step.get("depends_on", [])
            if depends_on:
                for dep in depends_on:
                    if dep not in step_id_map:
                        raise ValueError(f"Step {step['id']} depends_on unknown step id: {dep}")
        # Optionally: check for cycles in the DAG
        # ...

    def plan(self, goal: str, context: Dict[str, Any], episode_id: str = None) -> Dict[str, Any]:
        """
        Use LLM to generate a plan (sequence of tool calls) given a high-level goal and context.
        Returns a dict: {steps: [{tool, args}]}
        """
        # Load schema and format for prompt
        schema, valid_sources = self._load_schema()
        schema_str = self._format_schema_for_prompt(schema, valid_sources)
        # Load prompt templates
        prompts_path = Path(__file__).parent.parent.parent / "prompts.yaml"
        with open(prompts_path, "r") as f:
            prompts = yaml.safe_load(f)
        system_prompt = prompts["system"]
        user_prompt_template = prompts["user"]
        # Usage examples for analytics tools
        eid = episode_id or "EP-XXX"
        usage_examples = {
            "select_columns": "select_columns(df, ['season', 'player', 'points'])",
            "filter_rows": "filter_rows(df, 'points > 20')",
            "groupby_aggregate": "groupby_aggregate(df, ['season'], {'points': 'mean'})",
            "sort_values": "sort_values(df, by=['season'], ascending=[True])",
            "top_n": "top_n(df, groupby_cols=['season'], sort_col='points', n=1)",
            "merge": "merge(df1, df2, on=['player_id'], how='inner')",
            "pivot_table": "pivot_table(df, index=['season'], columns=['team'], values='points', aggfunc='mean')",
            "plot_time_series": f"plot_time_series(df, x='season', y='points', title='Points Over Time', episode_id='{eid}', hue='segment')",
            "describe": "describe(df)",
            "value_counts": "value_counts(df, column='team')",
        }
        tool_descriptions = []
        for name, entry in self.tools.items():
            fn = entry["fn"]
            args = entry["args"]
            doc = fn.__doc__.strip() if fn.__doc__ else ""
            example = usage_examples.get(name, None)
            desc = f"- {name}: {fn.__module__}.{fn.__name__}({', '.join(str(a) for a in args)})"
            if doc:
                desc += f"\n  Description: {doc}"
            if example:
                desc += f"\n  Example: {example}"
            tool_descriptions.append(desc)
        # Inject schema and episode_id into context
        context = context.copy() if context else {}
        context["schema"] = schema_str
        if episode_id:
            context["episode_id"] = episode_id
        user_prompt = user_prompt_template.format(
            goal=goal,
            tools="\n".join(tool_descriptions),
            context=json.dumps(context),
        )
        # Add explicit output instructions and format hints for reasoning models
        user_prompt += (
            "\nCRITICAL: Step Ordering and Column Creation Rules\n"
            "================================================\n"
            "- You MUST create columns (with assign_column, groupby_aggregate, etc.) before using them in any other step.\n"
            "- Steps must be ordered so that all dependencies are satisfied before use.\n"
            "- Never reference a column before it is created.\n"
            "- Do NOT create cycles or self-referential dependencies (e.g., a step that requires a column it creates).\n"
            "- If a step needs a column, ensure the step that creates it comes earlier in the plan.\n"
            "- Example of correct step ordering:\n"
            "  - id: assign_signup_flag\n"
            "    tool: assign_column\n"
            "    args:\n"
            "      df: load_data\n"
            "      column: signup_flag\n"
            "      expr: df['paid_signup_date'].notnull().astype(int)\n"
            "  - id: groupby_signup\n"
            "    tool: groupby_aggregate\n"
            "    args:\n"
            "      df: assign_signup_flag\n"
            "      groupby_cols: [treatment]\n"
            "      agg_dict: {signup_flag: {signups: 'sum'}}\n"
            "- In this example, 'signup_flag' is created before it is used in groupby_aggregate.\n"
            "- If you need to use a column in multiple steps, ensure it is created once and referenced by all later steps.\n"
            "- If you are unsure, err on the side of creating columns early and referencing them later.\n"
        )
        user_prompt += (
            "\nCRITICAL: Column Naming Rules for groupby_aggregate\n"
            "===================================================\n"
            "When using groupby_aggregate, the output column names follow these rules:\n"
            "1. Simple aggregation: {col: 'aggfunc'} → {col}_{aggfunc}\n"
            "   Example: {'score': 'mean'} → score_mean\n"
            "2. Named aggregation: {col: {'name': 'aggfunc'}} → {name}\n"
            "   Example: {'score': {'avg_score': 'mean'}} → avg_score\n"
            "   Example: {'signup_flag': {'signup_count': 'sum'}} → signup_count\n"
            "3. Multiple aggregations: {col: ['agg1', 'agg2']} → {col}_{agg1}, {col}_{agg2}\n"
            "   Example: {'score': ['mean', 'max']} → score_mean, score_max\n\n"
            "IMPORTANT: You MUST use the exact output column names in subsequent steps.\n"
            "Example workflow:\n"
            "1. groupby_aggregate with {'signup_flag': {'signups': 'sum'}, 'user_id': {'total_users': 'count'}}\n"
            "2. This creates columns: 'signups' and 'total_users'\n"
            "3. Use these exact names in subsequent steps\n\n"
            "IMPORTANT FOR REASONING MODELS:\n"
            "- Your output MUST be a valid YAML block, and must begin with 'steps:' on the first line.\n"
            "- Do NOT include any explanations, markdown code fencing, or extra text before or after the YAML block.\n"
            "- Do NOT include '```yaml' or '```' anywhere in your response.\n"
            "- The YAML must be directly parseable by PyYAML. Do not use tabs for indentation; use spaces only.\n"
            "- Do NOT include any commentary, reasoning, or summary outside the YAML block.\n"
            "- The output must be a single YAML object with a top-level 'steps:' key, whose value is a list of steps as described above.\n"
            "- If you are unsure, output only the YAML block and nothing else.\n"
        )
        user_prompt += (
            "\nIMPORTANT:\n"
            "- Output the plan as a YAML list of steps, not JSON.\n"
            "- Each step must have a unique id, a tool, an args dictionary, and (optionally) a depends_on list of step ids.\n"
            "- Use step ids as references in args (e.g., df: load_teams).\n"
            "- Only use columns that are present in the schema, or that you have created in a previous step using assign_column.\n"
            "- If you need a derived column (e.g., 'active_years'), you MUST create it using assign_column before using it in any other tool.\n"
            "- When creating new columns with assign_column, always apply subsequent assign_column steps to the output of the previous assign_column, so all new columns are accumulated.\n"
            "- When merging DataFrames, ensure all columns needed for downstream steps (e.g., for plotting) are preserved in the output.\n"
            "- Use concat to stack DataFrames vertically (e.g., after renaming columns to match), and use merge to join DataFrames on keys.\n"
            "- After groupby_aggregate, columns are flattened with specific naming patterns:\n"
            "  * For simple aggregations: {col: 'aggfunc'} → {col}_{aggfunc}\n"
            "    Example: {'score': 'mean'} → score_mean\n"
            "  * For named aggregations: {col: {'name': 'aggfunc'}} → {name}\n"
            "    Example: {'score': {'avg_score': 'mean'}} → avg_score\n"
            "    Example: {'signup_flag': {'signup_count': 'sum'}} → signup_count\n"
            "  * For multiple aggregations: {col: ['agg1', 'agg2']} → {col}_{agg1}, {col}_{agg2}\n"
            "    Example: {'score': ['mean', 'max']} → score_mean, score_max\n"
            "- CRITICAL: When using named aggregations, use the name you specified, not the original column name.\n"
            "  Example: If you do {'signup_flag': {'signup_count': 'sum'}}, use 'signup_count' in subsequent steps, NOT 'signup_flag_sum'.\n"
            "- Always use these flattened column names in subsequent steps.\n"
            "- Example groupby_aggregate and subsequent usage:\n"
            "  - id: agg_step\n"
            "    tool: groupby_aggregate\n"
            "    args:\n"
            "      df: previous_step\n"
            "      groupby_cols: ['category']\n"
            "      agg_dict:\n"
            "        value: {'avg': 'mean', 'max': 'max'}\n"
            "        count: 'sum'\n"
            "- Always include a generate_report step at the end of the plan.\n"
            "- For binary columns like is_home, use integer values 1 and 0, not strings.\n"
            "- Never reference columns that do not exist in the schema or in previous steps.\n"
            "- All string values in the YAML must be single-line and must not contain unescaped newlines or control characters.\n"
            "- All table and source arguments must be quoted strings (e.g., table: 'game_summary', not table: game_summary).\n"
        )
        self.logger.info(f"System prompt sent to LLM: {system_prompt}")
        self.logger.info(f"User prompt sent to LLM: {user_prompt}")
        print("System prompt sent to LLM:", system_prompt)
        print("User prompt sent to LLM:", user_prompt)

        def try_llm(user_prompt):
            # Use o4-mini or self.model as appropriate for the LLM call
            response = openai.chat.completions.create(
                model="o4-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_completion_tokens=32768,
                reasoning_effort="high",
            )

            # response = openai.chat.completions.create(
            #     model="gpt-4.1",
            #     messages=[
            #         {"role": "system", "content": system_prompt},
            #         {"role": "user", "content": user_prompt},
            #     ],
            #     max_tokens=32768,
            #     temperature=0.2,
            # )
            content = response.choices[0].message.content

            # Remove markdown code fencing if present (for o4-mini and similar models)
            if content.strip().startswith("```"):
                content = re.sub(r"^```[a-zA-Z]*\n?", "", content.strip())
                content = re.sub(r"```$", "", content.strip())

            # Pre-parse check for unterminated strings or control characters
            lines = content.splitlines()
            for line in lines:
                # o4-mini and similar models may sometimes emit odd YAML, so check for unbalanced quotes
                if line.count('"') % 2 == 1 or line.count("'") % 2 == 1:
                    if "\n" in line or "\r" in line or "\t" in line:
                        self.logger.error(
                            f"LLM output contains unterminated string or control character: {repr(line)}"
                        )
                        raise RuntimeError(
                            f"LLM output contains unterminated string or control character: {repr(line)}"
                        )

            # Extract YAML block (o4-mini may add extra text, so try to extract the YAML steps block)
            yaml_block = content
            match = re.search(r"steps:\s*\n(?:\s*- .+\n?)+", content, re.DOTALL)
            if match:
                yaml_block = match.group(0)

            try:
                plan = yaml.safe_load(yaml_block)
            except Exception as e:
                self.logger.error(
                    f"LLM planning failed: {e}\nRaw LLM response (YAML): {repr(yaml_block)}\nRaw LLM response: {repr(content)}"
                )
                sys.stderr.write("[LLM PLAN FAILURE] YAML that failed to parse:\n")
                sys.stderr.write(str(yaml_block) + "\n")
                sys.stderr.flush()
                raise RuntimeError(
                    f"LLM planning failed: {e}\nRaw LLM response (YAML): {repr(yaml_block)}\nRaw LLM response: {repr(content)}"
                )

            # Print the full plan (YAML or dict) to the terminal before validation
            sys.stderr.write("[LLM PLAN OUTPUT]\n")
            sys.stderr.write(yaml.dump(plan, sort_keys=False, default_flow_style=False) + "\n")
            sys.stderr.flush()

            # Validate plan structure
            if (
                not isinstance(plan, dict)
                or "steps" not in plan
                or not isinstance(plan["steps"], list)
            ):
                self.logger.error(f"LLM plan YAML missing 'steps' list. Parsed: {repr(plan)}")
                sys.stderr.write("[LLM PLAN FAILURE] Plan missing 'steps' list:\n")
                sys.stderr.write(str(plan) + "\n")
                sys.stderr.flush()
                raise RuntimeError(f"LLM plan YAML missing 'steps' list. Parsed: {repr(plan)}")

            # PATCH: Always patch the plan before validation/execution
            plan = self._patch_plan(plan, schema, valid_sources, episode_id, goal)

            # Validate plan
            try:
                self._validate_plan(plan, schema, valid_sources, episode_id=episode_id)
            except Exception as e:
                self.logger.error(f"Plan validation failed: {e}\nPlan that failed validation:")
                sys.stderr.write("[LLM PLAN FAILURE] Plan that failed validation:\n")
                sys.stderr.write(yaml.dump(plan, sort_keys=False, default_flow_style=False) + "\n")
                sys.stderr.flush()
                raise
            # === Validate all step references point to existing steps and are created before use ===
            step_names = [f"step_{i}" for i in range(len(plan.get("steps", [])))]
            for i, step in enumerate(plan.get("steps", [])):
                args = step.get("args", {})
                for k, v in args.items():
                    if isinstance(v, str) and v.startswith("step_"):
                        try:
                            ref_idx = int(v.split("_")[1])
                        except Exception:
                            self.logger.error(
                                f"Invalid step reference format: {v} in step {i} ({step.get('tool')})."
                            )
                            raise RuntimeError(
                                f"Invalid step reference format: {v} in step {i} ({step.get('tool')})."
                            )
                        if ref_idx >= i or ref_idx < 0 or ref_idx >= len(plan["steps"]):
                            self.logger.error(
                                f"Invalid step reference: {v} in step {i} ({step.get('tool')}). Plan: {json.dumps(plan, indent=2)}"
                            )
                            raise RuntimeError(
                                f"Invalid step reference: {v} in step {i} ({step.get('tool')}). Plan: {json.dumps(plan, indent=2)}"
                            )

            if "steps" in plan and plan["steps"]:
                self.logger.info("LLM planning succeeded.")
                self.logger.info("LLM plan steps:")
                for i, step in enumerate(plan["steps"]):
                    tool = step.get("tool")
                    args = step.get("args", {})
                    self.logger.info(f"  Step {i+1}: {tool}")
                    for k, v in args.items():
                        self.logger.info(f"    {k}: {v}")

                # --- Mermaid diagram generation ---
                def plan_to_mermaid(plan):
                    def escape(s):
                        return (
                            str(s).replace('"', "&quot;").replace("<", "&lt;").replace(">", "&gt;")
                        )

                    lines = ["flowchart TD"]
                    step_ids = []
                    for i, step in enumerate(plan.get("steps", [])):
                        step_id = f"step_{i}"
                        step_ids.append(step_id)
                        tool = step.get("tool", "?")
                        args = step.get("args", {})
                        # Show only key params (truncate long lists)
                        param_strs = []
                        for k, v in args.items():
                            v_str = str(v)
                            if isinstance(v, list) and len(v) > 5:
                                v_str = f"[{', '.join(map(str, v[:3]))}, ...]"
                            elif isinstance(v, str) and len(v) > 30:
                                v_str = v[:27] + "..."
                            param_strs.append(f"{k}={v_str}")
                        label = f"{tool}<br>{'<br>'.join(param_strs)}"
                        lines.append(f'    {step_id}["{escape(label)}"]')
                    # Add edges for step references
                    for i, step in enumerate(plan.get("steps", [])):
                        step_id = f"step_{i}"
                        args = step.get("args", {})
                        for v in args.values():
                            if isinstance(v, str) and v.startswith("step_"):
                                lines.append(f"    {v} --> {step_id}")
                            elif isinstance(v, list):
                                for vv in v:
                                    if isinstance(vv, str) and vv.startswith("step_"):
                                        lines.append(f"    {vv} --> {step_id}")
                    return "\n".join(lines)

                # Save Mermaid diagram to episode directory
                if episode_id:
                    ep_dir = Path("episodes") / episode_id
                    ep_dir.mkdir(parents=True, exist_ok=True)
                    mmd_path = ep_dir / "plan.mmd"
                    mmd_str = plan_to_mermaid(plan)
                    with open(mmd_path, "w") as f:
                        f.write(mmd_str)
                    # Also save as markdown for easy viewing
                    md_path = ep_dir / "plan.md"
                    with open(md_path, "w") as f:
                        f.write("```mermaid\n" + mmd_str + "\n```")
                # --- End Mermaid diagram generation ---
                return plan

        try:
            return try_llm(user_prompt)
        except Exception as e:
            self.logger.error(f"LLM planning failed: {e}. No valid plan generated. Aborting.")
            raise RuntimeError(f"LLM planning failed: {e}")

    def apply_plan(self, plan: Dict[str, Any], notebook_path: str) -> None:
        """
        Edit the notebook to insert code/markdown cells per the plan.
        """
        if plan is None:
            self.logger.error("No plan provided to apply_plan. Skipping notebook editing.")
            print("[ERROR] No plan provided to apply_plan. Skipping notebook editing.")
            return
        nb = nbformat.read(notebook_path, as_version=4)
        step_vars = {}
        for step in plan["steps"]:
            tool = self.tools[step["tool"]]
            var_name = step["id"]
            args = step["args"].copy()

            # Replace step id references in args with variable names recursively
            def replace_step_refs(val):
                if isinstance(val, str) and val in step_vars:
                    return step_vars[val]
                elif isinstance(val, list):
                    return [replace_step_refs(item) for item in val]
                elif isinstance(val, dict):
                    return {k: replace_step_refs(v) for k, v in val.items()}
                else:
                    return val

            args = {k: replace_step_refs(v) for k, v in args.items()}
            if step["tool"] == "set_sanity_passed":
                code = "sanity_passed = True\nprint('✅ Data sanity checks passed.')"
                notebook_editor.add_code(nb, code, tag=step["tool"])
            else:
                code = self._generate_code(tool["fn"], args, var_name=var_name, step_vars=step_vars)
                notebook_editor.add_code(nb, code, tag=step["tool"])
            step_vars[step["id"]] = var_name
        # Always append a final cell to emit sanity_passed as an execute_result for extract_globals
        globals_code = "{'sanity_passed': globals().get('sanity_passed', False)}"
        notebook_editor.add_code(nb, globals_code, tag="print_globals")
        nbformat.write(nb, notebook_path)

    def _make_hashable(self, obj):
        if isinstance(obj, list):
            return tuple(self._make_hashable(x) for x in obj)
        if isinstance(obj, dict):
            return tuple(sorted((k, self._make_hashable(v)) for k, v in obj.items()))
        return obj

    def _is_step_reference(self, v):
        return isinstance(v, str) and v.startswith("step_") and v[5:].isdigit()

    def _generate_code(self, fn, args: dict, var_name: str = "result", step_vars=None) -> str:
        """Generate Python code for a tool call.

        Args:
            fn: The function to call
            args: Dictionary of arguments
            var_name: Name of the variable to store the result
            step_vars: Dictionary mapping step IDs to variable names

        Returns:
            Generated Python code as a string
        """
        if step_vars is None:
            step_vars = {}

        # Get the module and function name
        module_name = fn.__module__
        fn_name = fn.__name__

        # Generate import statement
        code = f"from {module_name} import {fn_name}\n"

        # Convert args to Python code
        arg_strs = []
        for k, v in args.items():
            if isinstance(v, str):
                # If it's a step reference, use the variable name (no quotes)
                if v in step_vars or v in [step for step in step_vars.keys()]:
                    arg_strs.append(f"{k}={v}")
                else:
                    # If it's a pandas expression, evaluate it
                    if "[" in v and "]" in v and "." in v:
                        df_name = v.split("[")[0].strip()
                        if df_name in step_vars:
                            expr = v.replace(df_name, step_vars[df_name])
                            arg_strs.append(f"{k}={expr}")
                        else:
                            arg_strs.append(f"{k}={repr(v)}")
                    else:
                        arg_strs.append(f"{k}={repr(v)}")
            elif isinstance(v, list):
                list_items = []
                for item in v:
                    if isinstance(item, str) and (
                        item in step_vars or item in [step for step in step_vars.keys()]
                    ):
                        list_items.append(item)
                    else:
                        list_items.append(repr(item))
                arg_strs.append(f"{k}=[{', '.join(list_items)}]")
            elif isinstance(v, dict):
                dict_items = []
                for dk, dv in v.items():
                    if isinstance(dv, str) and (
                        dv in step_vars or dv in [step for step in step_vars.keys()]
                    ):
                        dict_items.append(f"{repr(dk)}: {dv}")
                    else:
                        dict_items.append(f"{repr(dk)}: {repr(dv)}")
                arg_strs.append(f"{k}={{{', '.join(dict_items)}}}")
            else:
                arg_strs.append(f"{k}={repr(v)}")

        code += f"{var_name} = {fn_name}({', '.join(arg_strs)})\n"
        code += f"print({var_name})\n"
        return code

    def run_episode(self, episode_id: str, max_iter: int = 3) -> None:
        """
        Main loop: plan, apply, execute, check for success, repeat up to max_iter.
        """
        # Load the episode metadata
        episode_path = Path(f"episodes/{episode_id}/episode.json")
        episode = load_episode(str(episode_path))
        goal = getattr(episode, "goal_statement", None) or getattr(episode, "title", None)
        if not goal:
            raise ValueError("No goal found in episode metadata.")
        notebook_path = Path(f"episodes/{episode_id}/analysis.ipynb")
        # Always clear the notebook at the start of the episode
        nb = nbformat.v4.new_notebook()
        nb.metadata["kernelspec"] = {
            "name": "python3",
            "display_name": "Python 3",
            "language": "python",
        }
        nbformat.write(nb, notebook_path)
        print("[INFO] Notebook cleared at start of episode.")
        self._tool_cache = {}  # (tool_name, args_tuple) -> result
        for i in range(max_iter):
            self.logger.info(f"Iteration {i+1}/{max_iter}")
            try:
                plan = self.plan(goal, context={}, episode_id=episode_id)
            except Exception as e:
                self.logger.error(f"Planning failed: {e}")
                self.logger.error("Episode terminated due to planning failure.")
                # Clear the notebook to avoid running stale/invalid code
                nb = nbformat.v4.new_notebook()
                nb.metadata["kernelspec"] = {
                    "name": "python3",
                    "display_name": "Python 3",
                    "language": "python",
                }
                nbformat.write(nb, notebook_path)
                print("[INFO] Notebook cleared due to planning failure.")
                return
            # 2. Apply plan
            self.apply_plan(plan, str(notebook_path))
            # 3. Run notebook and collect outputs
            output_path = notebook_path.with_name(f"analysis_executed.ipynb")
            kernel_runner.run_notebook(notebook_path, parameters={}, output_path=output_path)
            # 4. Extract all step outputs
            step_outputs = {}
            nb = nbformat.read(output_path, as_version=4)
            for cell in nb.cells:
                if cell.cell_type == "code" and cell.outputs:
                    # Try to extract variable assignment from code
                    lines = cell.source.splitlines()
                    for line in lines:
                        if "=" in line and not line.strip().startswith("#"):
                            var = line.split("=")[0].strip()
                            # Find output in cell.outputs
                            for out in cell.outputs:
                                if hasattr(out, "text"):
                                    step_outputs[var] = out.text.strip()
                                elif hasattr(out, "data") and "text/plain" in out.data:
                                    step_outputs[var] = out.data["text/plain"].strip()
            # 5. Find the generate_report step in the plan
            report_step = None
            for step in plan["steps"]:
                if step["tool"] == "generate_report":
                    report_step = step
                    break
            if report_step:
                # Collect all relevant step outputs for report
                report_kwargs = {}
                for dep in report_step.get("depends_on", []):
                    if dep in step_outputs:
                        report_kwargs[dep] = step_outputs[dep]
                # Also add outputs for known analysis tools if present
                for k in [
                    "correlation",
                    "regression",
                    "t_test",
                    "chi_square",
                    "anova",
                    "custom_metric",
                ]:
                    for step in plan["steps"]:
                        if step["tool"] == k and step["id"] in step_outputs:
                            report_kwargs[k + "_result"] = step_outputs[step["id"]]
                # Call generate_report with all collected results
                from ds_agent.analysis.analytics_tools import generate_report

                generate_report(goal=goal, episode_id=episode_id, **report_kwargs)
            # 6. Check for success
            globals_ = kernel_runner.extract_globals(output_path)
            print("DEBUG: Extracted globals:", globals_)
            print("DEBUG: sanity_passed value:", globals_.get("sanity_passed"))
            print("DEBUG: sanity_passed type:", type(globals_.get("sanity_passed")))
            if bool(globals_.get("sanity_passed")):
                self.logger.info("Sanity checks passed. Episode complete.")
                break
            else:
                self.logger.info("Sanity checks not passed. Continuing...")
        else:
            self.logger.warning("Max iterations reached. Episode incomplete.")

    def _patch_plan(
        self, plan: Dict[str, Any], schema: dict, valid_sources: set, episode_id: str, goal: str
    ) -> Dict[str, Any]:
        """Patch and validate the plan to ensure column references are correct.

        This method:
        1. Tracks available columns for each step
        2. Maintains DataFrame dependencies
        3. Tracks column lineage (which step created each column)
        4. Automatically fixes DataFrame references for missing columns
        5. Propagates columns forward through the plan
        6. Automatically reorders steps to satisfy dependencies
        7. Auto-inserts assign_column steps for missing columns if they exist in the original table
        8. Enhanced error reporting and suggestions
        """
        step_columns = {}
        df_dependencies = {}
        original_columns = set()
        column_creation = {}
        step_dependencies = {}
        agg_outputs = {}
        patched_steps = []
        steps_to_add = []
        column_lineage = {}  # column_name -> step_id
        step_ids = [step["id"] for step in plan.get("steps", [])]
        step_map = {step["id"]: step for step in plan.get("steps", [])}
        assign_column_counter = 0

        # First pass: identify all column creation steps and their dependencies
        for step in plan.get("steps", []):
            step_id = step["id"]
            tool = step.get("tool")
            args = step.get("args", {})
            depends_on = step.get("depends_on", [])
            step_dependencies[step_id] = set(depends_on)

            if tool == "assign_column":
                new_col = args.get("column")
                if new_col:
                    column_creation[new_col] = step_id
                    column_lineage[new_col] = step_id
                    if "df" in args:
                        step_dependencies[step_id].add(args["df"])
            elif tool == "groupby_aggregate":
                agg_dict = args.get("agg_dict", {})
                output_cols = set(args.get("groupby_cols", []))
                for col, agg in agg_dict.items():
                    if isinstance(agg, dict):
                        for name in agg.keys():
                            output_cols.add(name)
                            column_creation[name] = step_id
                            column_lineage[name] = step_id
                    elif isinstance(agg, list):
                        for a in agg:
                            col_name = f"{col}_{a}"
                            output_cols.add(col_name)
                            column_creation[col_name] = step_id
                            column_lineage[col_name] = step_id
                    else:
                        col_name = f"{col}_{agg}"
                        output_cols.add(col_name)
                        column_creation[col_name] = step_id
                        column_lineage[col_name] = step_id
                agg_outputs[step_id] = output_cols
                if "df" in args:
                    step_dependencies[step_id].add(args["df"])

        # Build dependency graph for topological sort
        col_needed_by_step = {}  # step_id -> set of needed columns
        for step in plan.get("steps", []):
            step_id = step["id"]
            tool = step.get("tool")
            args = step.get("args", {})
            needed_cols = set()
            for col_key in [
                "x",
                "y",
                "hue",
                "metric_col",
                "dim_col",
                "group_col",
                "value_col",
                "column",
            ]:
                if col_key in args and isinstance(args[col_key], str):
                    needed_cols.add(args[col_key])
            col_needed_by_step[step_id] = needed_cols

        # Topological sort: ensure all columns are created before use
        ordered_steps = []
        placed_steps = set()
        steps_left = set(step_ids)
        max_iters = len(step_ids) * 2
        iters = 0
        # Track the most recent step that contains each original column
        last_step_with_col = {}
        # Find the load_table step and its columns
        load_table_step = None
        for step in plan.get("steps", []):
            if step["tool"] == "load_table":
                load_table_step = step
                table = step["args"].get("table")
                if table in schema:
                    original_columns = set(schema[table]["columns"].keys())
        while steps_left and iters < max_iters:
            iters += 1
            progress = False
            for step_id in list(steps_left):
                needed_cols = col_needed_by_step.get(step_id, set())
                # All needed columns must be either in original columns or created by a placed step
                all_satisfied = True
                missing_cols = []
                for col in needed_cols:
                    if col in original_columns:
                        # Track the most recent step that contains this column
                        last_step_with_col[col] = load_table_step["id"] if load_table_step else None
                        continue
                    creator = column_lineage.get(col)
                    if creator and creator in placed_steps:
                        last_step_with_col[col] = creator
                        continue
                    if creator and creator not in placed_steps:
                        all_satisfied = False
                        break
                    if not creator:
                        missing_cols.append(col)
                        all_satisfied = False
                if all_satisfied:
                    ordered_steps.append(step_map[step_id])
                    placed_steps.add(step_id)
                    steps_left.remove(step_id)
                    progress = True
                elif missing_cols:
                    # Try to auto-insert assign_column steps for missing columns if they exist in original_columns
                    for col in missing_cols:
                        if col in original_columns:
                            # Insert assign_column step to propagate this column
                            assign_column_counter += 1
                            new_step_id = f"auto_assign_{col}_{assign_column_counter}"
                            # Use the most recent step that contains this column, or load_table
                            input_df = last_step_with_col.get(
                                col, load_table_step["id"] if load_table_step else None
                            )
                            new_step = {
                                "id": new_step_id,
                                "tool": "assign_column",
                                "args": {"df": input_df, "column": col, "expr": f"df['{col}']"},
                                "depends_on": [input_df] if input_df else [],
                            }
                            # Insert before the current step
                            step_map[new_step_id] = new_step
                            col_needed_by_step[new_step_id] = set([col])
                            column_creation[col] = new_step_id
                            column_lineage[col] = new_step_id
                            # Place the new step now
                            ordered_steps.append(new_step)
                            placed_steps.add(new_step_id)
                            steps_left.add(new_step_id)
                            last_step_with_col[col] = new_step_id
                            self.logger.info(
                                f"Auto-inserted assign_column step '{new_step_id}' for missing column '{col}' before step '{step_id}'."
                            )
                            progress = True
                        else:
                            self.logger.error(
                                f"Step {step_id}: Column '{col}' is referenced but never created and does not exist in the original table. Suggest inserting an assign_column step before this step."
                            )
                            raise ValueError(
                                f"Step {step_id}: Column '{col}' is referenced but never created and does not exist in the original table. Suggest inserting an assign_column step before this step."
                            )
            if not progress:
                # Could not make progress, must be a cycle or unsatisfiable dependency
                missing = {
                    sid: [
                        col
                        for col in col_needed_by_step[sid]
                        if (
                            col not in original_columns
                            and (
                                not column_lineage.get(col)
                                or column_lineage.get(col) not in placed_steps
                            )
                        )
                    ]
                    for sid in steps_left
                }
                self.logger.error(
                    f"Plan has unsatisfiable dependencies or a cycle. Steps left: {steps_left}. Missing columns: {missing}"
                )
                raise ValueError(
                    f"Plan has unsatisfiable dependencies or a cycle. Steps left: {steps_left}. Missing columns: {missing}"
                )

        # Now process steps in order, propagate columns, and fix references
        for idx, step in enumerate(ordered_steps):
            step_id = step["id"]
            tool = step.get("tool")
            args = step.get("args", {})

            # Track original columns from load_table
            if tool == "load_table":
                table = args.get("table")
                if table in schema:
                    original_columns = set(schema[table]["columns"].keys())
                    step_columns[step_id] = set(original_columns)
                    df_dependencies[step_id] = step_id
                    self.logger.info(f"Step {step_id}: Loaded columns {original_columns}")
            # Propagate columns from dependency
            elif "df" in args:
                df_ref = args["df"]
                if df_ref in step_columns:
                    step_columns[step_id] = set(step_columns[df_ref])
                    df_dependencies[step_id] = df_dependencies.get(df_ref, df_ref)
                    self.logger.info(f"Step {step_id}: Inherited columns from {df_ref}")
                else:
                    step_columns[step_id] = set()
            else:
                step_columns[step_id] = set()

            # Add/propagate columns for assign_column
            if tool == "assign_column":
                new_col = args.get("column")
                if new_col:
                    step_columns[step_id].add(new_col)
                    column_lineage[new_col] = step_id
                    self.logger.info(f"Step {step_id}: Created column {new_col}")
                df_dependencies[step_id] = args.get("df")
            # Add/propagate columns for groupby_aggregate
            elif tool == "groupby_aggregate":
                agg_dict = args.get("agg_dict", {})
                output_cols = set(args.get("groupby_cols", []))
                for col, agg in agg_dict.items():
                    if isinstance(agg, dict):
                        output_cols.update(agg.keys())
                        for name in agg.keys():
                            column_lineage[name] = step_id
                    elif isinstance(agg, list):
                        for a in agg:
                            col_name = f"{col}_{a}"
                            output_cols.add(col_name)
                            column_lineage[col_name] = step_id
                    else:
                        col_name = f"{col}_{agg}"
                        output_cols.add(col_name)
                        column_lineage[col_name] = step_id
                step_columns[step_id] = output_cols
                df_dependencies[step_id] = args.get("df")
                self.logger.info(f"Step {step_id}: Created columns {output_cols}")

            # --- Automatic DataFrame Reference Correction ---
            for col_key in [
                "x",
                "y",
                "hue",
                "metric_col",
                "dim_col",
                "group_col",
                "value_col",
                "column",
            ]:
                if col_key in args and isinstance(args[col_key], str):
                    col_name = args[col_key]
                    if col_name not in step_columns[step_id]:
                        creator = column_lineage.get(col_name)
                        if creator and creator != step_id:
                            creator_idx = (
                                [s["id"] for s in ordered_steps].index(creator)
                                if creator in [s["id"] for s in ordered_steps]
                                else -1
                            )
                            if creator_idx != -1 and creator_idx < idx:
                                self.logger.info(
                                    f"Step {step_id}: Column '{col_name}' not found, updating df to step '{creator}' that created it."
                                )
                                args["df"] = creator
                                step_columns[step_id] = set(step_columns[creator])
                                df_dependencies[step_id] = creator
                            else:
                                self.logger.error(
                                    f"Step {step_id}: Column '{col_name}' created in a future step '{creator}'. Plan is invalid."
                                )
                                raise ValueError(
                                    f"Step {step_id}: Column '{col_name}' created in a future step '{creator}'. Plan is invalid."
                                )
                        else:
                            self.logger.error(
                                f"Step {step_id}: Column '{col_name}' is referenced but never created and does not exist in the original table. Suggest inserting an assign_column step before this step."
                            )
                            raise ValueError(
                                f"Step {step_id}: Column '{col_name}' is referenced but never created and does not exist in the original table. Suggest inserting an assign_column step before this step."
                            )
            patched_steps.append(step)

        if steps_to_add:
            patched_steps.extend(steps_to_add)
            self.logger.info(f"Added {len(steps_to_add)} new steps to the plan")

        plan["steps"] = patched_steps
        return plan
