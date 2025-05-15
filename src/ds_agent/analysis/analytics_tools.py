import pandas as pd
from typing import List, Dict, Any, Optional, Union
import matplotlib.pyplot as plt
from pathlib import Path
from ds_agent.analysis.visuals import save_fig
import logging
import openai
import os
import json
import numpy as np
import re
import statsmodels.api as sm
from scipy import stats
import keyword
from sklearn.preprocessing import LabelEncoder


def configure_logging():
    """Configure logging format for better readability."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


# Configure logging when module is imported
configure_logging()


def select_columns(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """
    Return a DataFrame with only the specified columns.

    Args:
        df: The input DataFrame.
        columns: List of columns to select.

    Returns:
        pd.DataFrame: DataFrame with only the specified columns.

    Raises:
        ValueError: If any column is not found in the DataFrame.
    """
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"Columns not found in DataFrame: {missing}")
    return df[columns]


def filter_rows(df: pd.DataFrame, condition: str) -> pd.DataFrame:
    """Filter rows based on a pandas query condition.

    Args:
        df: Input DataFrame
        condition: Query condition (e.g., 'age > 30', 'status == "active"')

    Returns:
        Filtered DataFrame
    """
    # Extract column names from condition, ignoring literals and Python keywords
    import keyword
    import re

    # Track if we're inside quotes to ignore string literals
    in_quotes = False
    quote_char = None
    current_word = ""
    referenced_cols = set()

    for char in condition:
        if char in ['"', "'"]:
            if not in_quotes:
                in_quotes = True
                quote_char = char
            elif char == quote_char:
                in_quotes = False
                quote_char = None
            continue

        if in_quotes:
            continue

        if char.isalnum() or char == "_":
            current_word += char
        else:
            if current_word:
                # Only add if it's not a Python keyword, not a number, and not a boolean literal
                if (
                    not keyword.iskeyword(current_word)
                    and not current_word.isdigit()
                    and current_word not in ["True", "False", "None"]
                ):
                    referenced_cols.add(current_word)
                current_word = ""
            current_word = ""

    # Check last word if exists
    if current_word and (
        not keyword.iskeyword(current_word)
        and not current_word.isdigit()
        and current_word not in ["True", "False", "None"]
    ):
        referenced_cols.add(current_word)

    # Validate columns exist
    missing = [col for col in referenced_cols if col not in df.columns]
    if missing:
        raise ValueError(
            f"Columns referenced in query are missing: {missing}. "
            f"Available columns: {list(df.columns)}. Query: {condition}"
        )

    return df.query(condition)


def groupby_aggregate(
    df: pd.DataFrame,
    groupby_cols: List[str],
    agg_dict: Dict[str, Union[str, List[str], Dict[str, str]]],
    dropna: bool = True,
    as_index: bool = False,
) -> pd.DataFrame:
    """
    Group by columns and aggregate using the provided aggregation dictionary.

    Args:
        df: The DataFrame to group and aggregate.
        groupby_cols: List of columns to group by.
        agg_dict: Dictionary specifying aggregations. Supports:
            - {col: 'aggfunc'} (e.g., {'score': 'mean'})
            - {col: ['aggfunc1', 'aggfunc2']} (e.g., {'score': ['mean', 'max']})
            - Named aggregations: {col: {'agg_name': 'aggfunc'}} (e.g., {'score': {'avg_score': 'mean'}})
        dropna: Whether to drop NA group keys. Default True.
        as_index: Whether to return groupby columns as index. Default False.

    Returns:
        pd.DataFrame: The aggregated DataFrame.

    Raises:
        ValueError: If aggregation parameters are invalid or aggregation fails.
    """
    try:
        logging.info(f"[groupby_aggregate] Input DataFrame columns: {list(df.columns)}")
        logging.info(f"[groupby_aggregate] Aggregation dict: {agg_dict}")

        # Create named aggregations to ensure consistent column names
        named_agg_dict = {}
        for col, agg in agg_dict.items():
            if isinstance(agg, str):
                named_agg_dict[col] = (col, agg)
            elif isinstance(agg, list):
                for agg_func in agg:
                    named_agg_dict[f"{col}_{agg_func}"] = (col, agg_func)
            elif isinstance(agg, dict):
                for agg_name, agg_func in agg.items():
                    named_agg_dict[agg_name] = (col, agg_func)

        grouped = df.groupby(groupby_cols, dropna=dropna, as_index=as_index)
        result = grouped.agg(**named_agg_dict)

        # Reset index if needed
        if not as_index:
            result = result.reset_index()

        logging.info(f"[groupby_aggregate] Output DataFrame columns: {list(result.columns)}")
        return result

    except Exception as e:
        raise ValueError(
            f"Groupby-aggregate failed: {e}\nParams: groupby_cols={groupby_cols}, agg_dict={agg_dict}"
        )


def sort_values(
    df: pd.DataFrame, by: List[str], ascending: Optional[List[bool]] = None
) -> pd.DataFrame:
    """Sort the DataFrame by the specified columns."""
    if ascending is None:
        ascending = [True] * len(by)
    return df.sort_values(by=by, ascending=ascending)


def top_n(df: pd.DataFrame, groupby_cols: List[str], sort_col: str, n: int) -> pd.DataFrame:
    """Return the top N rows per group, or top N overall if groupby_cols is empty."""
    logging.info(f"[top_n] Available columns: {list(df.columns)}")
    logging.info(f"[top_n] DataFrame info:\n{df.info()}")
    if sort_col not in df.columns:
        raise ValueError(
            f"Sort column '{sort_col}' not found in DataFrame. Available columns: {list(df.columns)}"
        )
    sorted_df = df.sort_values(sort_col, ascending=False)
    if groupby_cols:
        return sorted_df.groupby(groupby_cols).head(n)
    else:
        return sorted_df.head(n)


def merge(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    on: Optional[Union[str, List[str]]] = None,
    how: str = "inner",
    left_on: Optional[Union[str, List[str]]] = None,
    right_on: Optional[Union[str, List[str]]] = None,
    suffixes: tuple = ("_x", "_y"),
    validate: Optional[str] = None,
    indicator: bool = False,
) -> pd.DataFrame:
    """
    Merge two DataFrames with flexible options (wrapper around pandas.merge).

    Args:
        df1: Left DataFrame.
        df2: Right DataFrame.
        on: Column(s) to join on. Must be present in both DataFrames. (Optional if using left_on/right_on)
        how: Type of merge: 'left', 'right', 'outer', 'inner'. Default is 'inner'.
        left_on: Column(s) from left DataFrame to join on. (Alternative to 'on')
        right_on: Column(s) from right DataFrame to join on. (Alternative to 'on')
        suffixes: Suffixes to apply to overlapping columns. Default ('_x', '_y').
        validate: If specified, checks if merge is of specified type (e.g., 'one_to_one', 'one_to_many').
        indicator: If True, adds a column to output DataFrame indicating source of each row.

    Returns:
        pd.DataFrame: The merged DataFrame.

    Raises:
        ValueError: If merge parameters are invalid or merge fails.
    """
    try:
        result = pd.merge(
            df1,
            df2,
            how=how,
            on=on,
            left_on=left_on,
            right_on=right_on,
            suffixes=suffixes,
            validate=validate,
            indicator=indicator,
        )
    except Exception as e:
        raise ValueError(
            f"Merge failed: {e}\nParams: on={on}, left_on={left_on}, right_on={right_on}, how={how}"
        )
    return result


def pivot_table(
    df: pd.DataFrame, index: List[str], columns: List[str], values: str, aggfunc: str = "mean"
) -> pd.DataFrame:
    """Create a pivot table from the DataFrame."""
    return pd.pivot_table(
        df, index=index, columns=columns, values=values, aggfunc=aggfunc
    ).reset_index()


def plot_time_series(
    df: pd.DataFrame,
    x: str,
    y: str,
    title: str,
    episode_id: str = "default",
    hue: str = None,
) -> str:
    """
    Plot a time series and save as PNG. Returns relative path to the image.
    If 'hue' is provided, plot a separate line for each value in the hue column.
    Ensures x-axis is datetime.

    Args:
        df: The input DataFrame.
        x: Column for x-axis (should be datetime or convertible).
        y: Column for y-axis (numeric).
        title: Plot title.
        episode_id: Used for saving the plot.
        hue: Optional column to group lines by.

    Returns:
        str: Relative path to the saved image.

    Raises:
        ValueError: If required columns are missing or DataFrame is empty.
    """
    logging.info(f"[plot_time_series] Starting plot with columns: {list(df.columns)}")
    logging.info(f"[plot_time_series] Requested columns - x: {x}, y: {y}, hue: {hue}")

    if df.empty:
        raise ValueError("Cannot plot time series on an empty DataFrame.")

    # Check for missing columns
    missing_cols = [col for col in [x, y] + ([hue] if hue else []) if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Columns not found in DataFrame: {missing_cols}. "
            f"Available columns: {list(df.columns)}"
        )

    fig, ax = plt.subplots(figsize=(10, 6))

    # Ensure x is datetime
    if not pd.api.types.is_datetime64_any_dtype(df[x]):
        try:
            df = df.copy()
            df[x] = pd.to_datetime(df[x], errors="coerce")
            logging.info(f"[plot_time_series] Converted {x} to datetime")
        except Exception as e:
            logging.error(f"[plot_time_series] Failed to convert {x} to datetime: {e}")
            raise ValueError(f"Column {x} must be convertible to datetime")

    # Plot the data
    if hue and hue in df.columns:
        for value, group in df.groupby(hue):
            ax.plot(group[x], group[y], marker="o", label=str(value))
        ax.legend(title=hue)
    else:
        ax.plot(df[x], df[y], marker="o")

    ax.set_title(title)
    ax.set_xlabel(x)
    ax.set_ylabel(y)

    # Save the plot
    path = save_fig(fig, episode_id, title.replace(" ", "_").lower())
    plt.close(fig)

    logging.info(f"[plot_time_series] Plot saved to {path}")
    return path


def plot_bar(
    df: pd.DataFrame, x: str, y: str = None, title: str = "", episode_id: str = "default"
) -> str:
    """
    Plot a bar chart and save as PNG. If y is None, plot value counts of x.
    """
    if df.empty:
        raise ValueError("Cannot plot bar chart on an empty DataFrame.")
    if x not in df.columns:
        raise ValueError(f"Column '{x}' not found in DataFrame.")
    import matplotlib.pyplot as plt
    from pathlib import Path

    fig, ax = plt.subplots(figsize=(10, 6))
    if y is None:
        counts = df[x].value_counts()
        ax.bar(counts.index.astype(str), counts.values)
        ax.set_ylabel("Count")
    else:
        if y not in df.columns:
            raise ValueError(f"Column '{y}' not found in DataFrame.")
        ax.bar(df[x], df[y])
        ax.set_ylabel(y)
    ax.set_title(title)
    ax.set_xlabel(x)
    path = save_fig(fig, episode_id, title.replace(" ", "_").lower())
    plt.close(fig)
    return path


def plot_scatter(df: pd.DataFrame, x: str, y: str, title: str, episode_id: str = "default") -> str:
    """
    Plot a scatter plot and save as PNG. Returns relative path to the image.

    Args:
        df: The input DataFrame.
        x: Column for x-axis (numeric).
        y: Column for y-axis (numeric).
        title: Plot title.
        episode_id: Used for saving the plot.

    Returns:
        str: Relative path to the saved image.

    Raises:
        ValueError: If required columns are missing or DataFrame is empty.
    """
    if df.empty:
        raise ValueError("Cannot plot scatter on an empty DataFrame.")
    for col in [x, y]:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame.")
        # Convert to numeric, coercing errors to NaN
        df[col] = pd.to_numeric(df[col], errors="coerce")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(df[x], df[y])
    ax.set_title(title)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    path = save_fig(fig, episode_id, title.replace(" ", "_").lower())
    plt.close(fig)
    return path


def describe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return basic statistics for each column in the DataFrame.

    Args:
        df: The input DataFrame.

    Returns:
        pd.DataFrame: Descriptive statistics for all columns.

    Raises:
        ValueError: If the DataFrame is empty.
    """
    if df.empty:
        logging.warning("[describe] DataFrame is empty. Returning empty DataFrame.")
        return pd.DataFrame()
    return df.describe(include="all")


def value_counts(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Return value counts for a column as a DataFrame.

    Args:
        df: The input DataFrame.
        column: The column to count values for.

    Returns:
        pd.DataFrame: DataFrame with value counts for the column.

    Raises:
        ValueError: If the DataFrame is empty.
    """
    if df.empty:
        logging.warning(
            f"[value_counts] DataFrame is empty. Returning empty DataFrame for column '{column}'."
        )
        return pd.DataFrame({column: [], "count": []})
    if column not in df.columns:
        logging.warning(
            f"[value_counts] Column '{column}' not found in DataFrame. Returning empty DataFrame."
        )
        return pd.DataFrame({column: [], "count": []})
    return df[column].value_counts().reset_index(name="count").rename(columns={"index": column})


def set_sanity_passed() -> None:
    """Set the global variable sanity_passed = True to indicate successful analysis completion."""
    global sanity_passed
    sanity_passed = True


def rename_columns(df: pd.DataFrame, rename_dict: Dict[str, str]) -> pd.DataFrame:
    """
    Rename columns in a DataFrame.

    Args:
        df: The input DataFrame.
        rename_dict: Dictionary mapping old column names to new names.

    Returns:
        pd.DataFrame: DataFrame with columns renamed.

    Raises:
        ValueError: If any old column is not found in the DataFrame.
    """
    missing = [col for col in rename_dict if col not in df.columns]
    if missing:
        raise ValueError(f"Columns to rename not found in DataFrame: {missing}")
    return df.rename(columns=rename_dict)


def assign_column(df: pd.DataFrame, column: str, expr: str) -> pd.DataFrame:
    """
    Assign a new column to the DataFrame based on a pandas eval/where expression or Python expression.

    Args:
        df: The input DataFrame.
        column: Name of the new or existing column to assign.
        expr: Expression to evaluate. Can use 'df' (the DataFrame) and 'np' (numpy).
            Example: "np.where((df['a'] > 0), 'pos', 'neg')"

    Returns:
        pd.DataFrame: DataFrame with the new/updated column.

    Raises:
        ValueError: If the expression fails to evaluate.
    """
    df = df.copy()
    try:
        df[column] = eval(expr, {"df": df, "np": np, "pd": pd})
    except Exception as e:
        raise ValueError(f"Failed to assign column '{column}' with expr '{expr}': {e}")
    return df


def generate_report(
    goal: str,
    top_n_table: pd.DataFrame = None,
    value_counts: dict = None,
    describe_stats: pd.DataFrame = None,
    segmentation: str = None,
    baseline_describe_stats: pd.DataFrame = None,
    baseline_value_counts: dict = None,
    extra_segmentations: dict = None,
    top_n_value: int = 20,
    episode_id: str = None,
    **kwargs,
) -> str:
    """
    Generate a markdown report answering the analysis goal using the LLM.
    Accepts arbitrary additional analysis results via **kwargs.
    """
    import json
    import pandas as pd
    import logging

    logging.basicConfig(level=logging.INFO)
    logging.info("[generate_report] Starting report generation")
    logging.info(f"[generate_report] Goal: {goal}")
    logging.info(f"[generate_report] Episode ID: {episode_id}")

    def render_section(name, value):
        """Render a section for the LLM prompt based on the value type."""
        if value is None:
            return ""
        header = f"## {name.replace('_', ' ').title()}\n"
        if isinstance(value, pd.DataFrame):
            return header + value.to_markdown(index=False) + "\n"
        elif isinstance(value, (tuple, list)):
            return header + ", ".join(str(x) for x in value) + "\n"
        elif isinstance(value, dict):
            return header + json.dumps(value, indent=2) + "\n"
        else:
            return header + str(value) + "\n"

    # Compose dynamic sections from kwargs
    dynamic_sections = ""
    for k, v in kwargs.items():
        dynamic_sections += render_section(k, v)

    # Compose legacy sections for backward compatibility
    legacy_sections = ""
    legacy_sections += render_section("Top N Table", top_n_table)
    legacy_sections += render_section("Value Counts", value_counts)
    legacy_sections += render_section("Describe Stats", describe_stats)
    legacy_sections += render_section("Segmentation", segmentation)
    legacy_sections += render_section("Baseline Describe Stats", baseline_describe_stats)
    legacy_sections += render_section("Baseline Value Counts", baseline_value_counts)
    legacy_sections += render_section("Extra Segmentations", extra_segmentations)

    # Compose prompt
    prompt = f"""
You are a data science assistant. Write a concise, insightful markdown report that answers the following analysis goal.

# Goal
{goal}

# Analysis Results
{dynamic_sections}{legacy_sections}

Instructions:
- Synthesize findings from all analyses above.
- The report needs to be highly polished and intended to convey to the reader both technical proficiency and business acumen.  The report needs to have a narrative style, not simply bullet points. It should start with a stronger narrative leaning at the beginning of the doc in the executive summary and become increasingly more scientific and academic as the explanation details increase.   
- The expected format of the report is as follows:
    1. Executive Summary - a high-level summary concisely conveying the key insights and recommendations. Intended audience: Executives, PMs
    2. Key Insights - a more detailed summary of the key insights. Intended audience: PMs, Engineers, Data Scientists
    3. Recommendations - a summary of the recommendations. Intended audience: PMs, Engineers, Data Scientists. Details - a detailed explanation of the analysis and the assumptions made. Is the section where you go into the nitty gritty of the analysis and to `show your work`. The reader must be able to verify all of your work based on the data and analysis you provide. There must be no ambiguity in your analysis. Intended audience: Data Scientists
    5. Appendix - a list of assumptions and any other relevant information. Intended audience: Data Scientists
- Reference statistical results and explain their meaning.
- Write a clear, professional markdown report.
- Utilize visualizations to support your analysis. Insert them at the appropriate points in the report with Markdown tags.
"""

    # Log prompt length and warn if very long
    prompt_length = len(prompt)
    logging.info(f"[generate_report] Prompt length (characters): {prompt_length}")
    if prompt_length > 32000:
        logging.warning(
            f"[generate_report] Prompt is very long (>32,000 chars). This may cause LLM truncation or empty responses."
        )

    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if not openai_api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable not set.")

    try:
        logging.info("[generate_report] Making API call to OpenAI...")
        response = openai.chat.completions.create(
            model="o4-mini",
            messages=[
                {"role": "system", "content": "You are a data science assistant."},
                {"role": "user", "content": prompt},
            ],
            max_completion_tokens=32768,
            reasoning_effort="high",
        )
        logging.info(f"[generate_report] API call successful. Response object: {response}")

        if not response.choices:
            logging.error("[generate_report] No choices in API response")
            return "Error: No choices in API response. Please try again."

        if not response.choices[0].message:
            logging.error("[generate_report] No message in first choice")
            return "Error: No message in first choice. Please try again."

        full_response = response.choices[0].message.content.strip()
        logging.info(f"[generate_report] Full API response length: {len(full_response)}")
        logging.info(f"[generate_report] First 200 chars of response: {full_response[:200]}")

        if not full_response:
            logging.error("[generate_report] Empty response from API")
            return (
                "Error: Empty response from API. This may be due to a prompt that is too long, "
                "model limitations, or a temporary service issue. Try reducing the Top-N value, "
                "simplifying the analysis goal, or rerunning the report. If the problem persists, "
                "check the logs for prompt length and content."
            )

        # Parse for --- Final Report --- delimiter
        if "--- Final Report ---" in full_response:
            reasoning, report = full_response.split("--- Final Report ---", 1)
            reasoning = reasoning.strip()
            report = report.strip()
            logging.info(f"[generate_report] Found delimiter. Report length: {len(report)}")
        else:
            logging.warning(
                "[generate_report] No delimiter found in response. Returning full response as report."
            )
            reasoning = ""
            report = full_response.strip()

        if not report:
            logging.error("[generate_report] Empty report after parsing")
            return "Error: Empty report after parsing. Please try again."

        logging.info(f"[generate_report] Final report length: {len(report)}")
        logging.info(f"[generate_report] First 200 chars of final report: {report[:200]}")

    except Exception as e:
        logging.error(f"[generate_report] API call failed: {str(e)}")
        return f"Error: API call failed - {str(e)}"

    # Save to episode directory if provided
    if episode_id:
        out_dir = Path("episodes") / episode_id
        out_dir.mkdir(parents=True, exist_ok=True)
        report_path = out_dir / "report.md"
        with open(report_path, "w") as f:
            f.write(report)
        # Optionally save reasoning step
        reasoning_path = out_dir / "report_reasoning.txt"
        with open(reasoning_path, "w") as f:
            f.write(reasoning)
        logging.info(f"[generate_report] Report saved to {report_path}")
        logging.info(f"[generate_report] Reasoning saved to {reasoning_path}")
    return report


def plot_histogram(df: pd.DataFrame, column: str, title: str, episode_id: str = "default") -> str:
    """
    Plot a histogram for a numeric column and save as PNG. Returns relative path to the image.

    Args:
        df: The input DataFrame.
        column: The column to plot (numeric).
        title: Plot title.
        episode_id: Used for saving the plot.

    Returns:
        str: Relative path to the saved image.

    Raises:
        ValueError: If the DataFrame is empty or column is missing/not numeric.
    """
    if df.empty:
        raise ValueError("Cannot plot histogram on an empty DataFrame.")
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame.")
    if not pd.api.types.is_numeric_dtype(df[column]):
        raise ValueError(f"Column '{column}' is not numeric and cannot be histogrammed.")
    import matplotlib.pyplot as plt
    from pathlib import Path

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(df[column].dropna(), bins=10, color="skyblue", edgecolor="black")
    ax.set_title(title)
    ax.set_xlabel(column)
    ax.set_ylabel("Frequency")
    plt.tight_layout()
    image_dir = Path("episodes") / episode_id / "charts"
    image_dir.mkdir(parents=True, exist_ok=True)
    img_path = image_dir / f"hist_{column}.png"
    fig.savefig(img_path)
    plt.close(fig)
    return str(img_path)


def concat(dfs, axis=0, ignore_index=True):
    """
    Concatenate a list of DataFrames along the given axis.

    Args:
        dfs: List of DataFrames to concatenate.
        axis: 0 for rows (vertical), 1 for columns (horizontal). Default 0.
        ignore_index: Whether to ignore index on concat. Default True.

    Returns:
        pd.DataFrame: Concatenated DataFrame.

    Raises:
        ValueError: If dfs is empty or DataFrames are incompatible for concat.
    """
    if not dfs or len(dfs) < 1:
        raise ValueError("No DataFrames provided to concat.")
    try:
        return pd.concat(dfs, axis=axis, ignore_index=ignore_index)
    except Exception as e:
        raise ValueError(f"Concat failed: {e}")


def correlation(df, col1, col2, method="pearson"):
    """
    Compute correlation between two columns using the specified method ('pearson', 'spearman', 'kendall').
    Args:
        df: Input DataFrame.
        col1: First column name.
        col2: Second column name.
        method: Correlation method (default 'pearson').
    Returns:
        Correlation coefficient (float).
    """
    if col1 not in df.columns or col2 not in df.columns:
        raise ValueError(f"Columns {col1}, {col2} not found in DataFrame.")
    if method not in ["pearson", "spearman", "kendall"]:
        raise ValueError("method must be one of 'pearson', 'spearman', 'kendall'")
    return df[[col1, col2]].corr(method=method).iloc[0, 1]


def regression(
    df: pd.DataFrame, y: str, X: list[str], model_type: str = "linear", params: dict = None
) -> dict:
    """
    Perform regression analysis (linear or logistic).

    Args:
        df: Input DataFrame
        y: Target variable column name
        X: List of predictor variable column names
        model_type: 'linear' or 'logistic'
        params: Additional parameters for the model

    Returns:
        Dictionary containing model results and statistics
    """

    # Validate input data types
    def validate_and_convert_column(col_data, col_name, is_target=False):
        if col_data.dtype == "object":
            # For target variable in logistic regression, convert to binary
            if is_target and model_type == "logistic":
                if col_data.nunique() != 2:
                    raise ValueError(
                        f"Target column {col_name} must be binary for logistic regression"
                    )
                le = LabelEncoder()
                return pd.Series(le.fit_transform(col_data), index=col_data.index)
            # For categorical predictors, use one-hot encoding
            else:
                return pd.get_dummies(col_data, prefix=col_name, drop_first=True)
        elif col_data.dtype == "bool":
            return col_data.astype(int)
        elif not np.issubdtype(col_data.dtype, np.number):
            raise ValueError(f"Column {col_name} must be numeric or categorical")
        return col_data

    # Validate and convert target variable
    yvec = validate_and_convert_column(df[y], y, is_target=True)

    # Validate and convert predictors
    X_dfs = []
    for x_col in X:
        x_data = validate_and_convert_column(df[x_col], x_col)
        if isinstance(x_data, pd.DataFrame):  # One-hot encoded
            X_dfs.append(x_data)
        else:
            X_dfs.append(pd.DataFrame({x_col: x_data}))

    # Combine all predictors
    Xmat = pd.concat(X_dfs, axis=1)

    # Add constant for linear regression
    if model_type == "linear":
        Xmat = sm.add_constant(Xmat)

    # Fit model
    if model_type == "linear":
        model = sm.OLS(yvec, Xmat)
    elif model_type == "logistic":
        model = sm.Logit(yvec, Xmat)
    else:
        raise ValueError("model_type must be 'linear' or 'logistic'")

    results = model.fit()

    # Prepare output
    output = {
        "model_type": model_type,
        "n_observations": len(df),
        "n_predictors": len(X),
        "r_squared": results.rsquared if model_type == "linear" else results.prsquared,
        "coefficients": results.params.to_dict(),
        "p_values": results.pvalues.to_dict(),
        "conf_int": results.conf_int().to_dict(),
        "aic": results.aic,
        "bic": results.bic,
    }

    # Add model-specific statistics
    if model_type == "logistic":
        output.update(
            {
                "log_likelihood": results.llf,
                "pseudo_r_squared": results.prsquared,
            }
        )

    return output


def t_test(df, group_col, value_col, group1, group2):
    """
    Perform a t-test comparing value_col between two groups in group_col.
    Args:
        df: Input DataFrame.
        group_col: Column to group by.
        value_col: Value column to compare.
        group1: First group value.
        group2: Second group value.
    Returns:
        t-statistic, p-value.
    """
    if group_col not in df.columns or value_col not in df.columns:
        raise ValueError("Group or value column not found in DataFrame.")
    vals1 = df[df[group_col] == group1][value_col].dropna()
    vals2 = df[df[group_col] == group2][value_col].dropna()
    tstat, pval = stats.ttest_ind(vals1, vals2, equal_var=False)
    return tstat, pval


def chi_square(df, col1, col2):
    """
    Perform a chi-square test of independence between two categorical columns.
    Args:
        df: Input DataFrame.
        col1: First categorical column.
        col2: Second categorical column.
    Returns:
        chi2 statistic, p-value.
    """
    if col1 not in df.columns or col2 not in df.columns:
        raise ValueError("Columns not found in DataFrame.")
    contingency = pd.crosstab(df[col1], df[col2])
    from scipy.stats import chi2_contingency

    chi2, p, dof, expected = chi2_contingency(contingency)
    return chi2, p


def anova(df, group_col, value_col):
    """
    Perform one-way ANOVA for value_col across groups in group_col.
    Args:
        df: Input DataFrame.
        group_col: Grouping column.
        value_col: Value column.
    Returns:
        F-statistic, p-value.
    """
    if group_col not in df.columns or value_col not in df.columns:
        raise ValueError("Group or value column not found in DataFrame.")
    groups = [g[value_col].dropna() for _, g in df.groupby(group_col)]
    fstat, pval = stats.f_oneway(*groups)
    return fstat, pval


def event_filter(df, event_col, event_values):
    """
    Filter DataFrame to rows where event_col is in event_values.
    Args:
        df: Input DataFrame.
        event_col: Event column name.
        event_values: List of values to keep.
    Returns:
        Filtered DataFrame.
    """
    if event_col not in df.columns:
        raise ValueError("Event column not found in DataFrame.")
    return df[df[event_col].isin(event_values)]


def time_window_filter(df, time_col, start, end):
    """
    Filter DataFrame to rows where time_col is between start and end (inclusive).
    Args:
        df: Input DataFrame.
        time_col: Time column name.
        start: Start time (string or datetime).
        end: End time (string or datetime).
    Returns:
        Filtered DataFrame.
    """
    if time_col not in df.columns:
        raise ValueError("Time column not found in DataFrame.")
    col = pd.to_datetime(df[time_col], errors="coerce")
    start = pd.to_datetime(start)
    end = pd.to_datetime(end)
    return df[(col >= start) & (col <= end)]


def custom_metric(df, metric_name, expr):
    """
    Compute a custom metric column using a pandas expression or Python code.
    Args:
        df: Input DataFrame.
        metric_name: Name of new metric column.
        expr: Expression to compute (string, evaluated with df context).
    Returns:
        DataFrame with new metric column.
    """
    df = df.copy()
    try:
        df[metric_name] = eval(expr, {"df": df, "np": np, "pd": pd})
    except Exception as e:
        raise ValueError(f"Failed to compute custom metric '{metric_name}' with expr '{expr}': {e}")
    return df


def robust_merge(dfs, join_keys, how="inner", suffixes=None, validate=None, indicator=False):
    """
    Perform a robust multi-table merge on join_keys, handling missing keys and suffixes.
    Args:
        dfs: List of DataFrames to merge.
        join_keys: List of join key(s).
        how: Merge type ('inner', 'outer', etc.).
        suffixes: Suffixes for overlapping columns.
        validate: Merge validation option.
        indicator: Add merge indicator column.
    Returns:
        Merged DataFrame.
    """
    if not dfs or len(dfs) < 2:
        raise ValueError("At least two DataFrames required for robust_merge.")
    result = dfs[0]
    for i, df in enumerate(dfs[1:], 1):
        sfx = suffixes[i - 1] if suffixes and len(suffixes) > i - 1 else (f"_x{i}", f"_y{i}")
        result = pd.merge(
            result,
            df,
            how=how,
            on=join_keys,
            suffixes=sfx,
            validate=validate,
            indicator=indicator if i == len(dfs) - 1 else False,
        )
    return result


def preprocess_for_regression(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: list[str],
    categorical_cols: list[str] = None,
    binary_cols: list[str] = None,
) -> pd.DataFrame:
    """
    Preprocess data for regression analysis.

    Args:
        df: Input DataFrame
        target_col: Name of the target variable column
        feature_cols: List of feature column names
        categorical_cols: List of categorical columns to one-hot encode
        binary_cols: List of binary columns to convert to 0/1

    Returns:
        Preprocessed DataFrame ready for regression
    """
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import LabelEncoder

    # Make a copy to avoid modifying the original
    processed_df = df.copy()

    # Convert binary columns to 0/1
    if binary_cols:
        for col in binary_cols:
            if col in processed_df.columns:
                if processed_df[col].dtype == "object":
                    le = LabelEncoder()
                    processed_df[col] = le.fit_transform(processed_df[col])
                elif processed_df[col].dtype == "bool":
                    processed_df[col] = processed_df[col].astype(int)

    # One-hot encode categorical columns
    if categorical_cols:
        for col in categorical_cols:
            if col in processed_df.columns:
                dummies = pd.get_dummies(processed_df[col], prefix=col, drop_first=True)
                processed_df = pd.concat([processed_df.drop(col, axis=1), dummies], axis=1)

    # Ensure target variable is numeric for logistic regression
    if processed_df[target_col].dtype == "object":
        le = LabelEncoder()
        processed_df[target_col] = le.fit_transform(processed_df[target_col])
    elif processed_df[target_col].dtype == "bool":
        processed_df[target_col] = processed_df[target_col].astype(int)

    # Validate all columns are numeric
    non_numeric = (
        processed_df[feature_cols + [target_col]].select_dtypes(include=["object"]).columns
    )
    if len(non_numeric) > 0:
        raise ValueError(f"Non-numeric columns found after preprocessing: {list(non_numeric)}")

    return processed_df


try:
    TOOL_REGISTRY
except NameError:
    TOOL_REGISTRY = {}

TOOL_REGISTRY["generate_report"] = {
    "fn": generate_report,
    "args": [
        "goal",
        "top_n_table",
        "value_counts",
        "describe_stats",
        "segmentation",
        "baseline_describe_stats",
        "baseline_value_counts",
        "extra_segmentations",
        "top_n_value",
        "episode_id",
    ],
    "description": generate_report.__doc__,
}

TOOL_REGISTRY["plot_histogram"] = {
    "fn": plot_histogram,
    "args": ["df", "column", "title", "episode_id"],
    "description": plot_histogram.__doc__,
}
