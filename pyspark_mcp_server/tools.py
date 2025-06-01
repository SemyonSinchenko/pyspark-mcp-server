from __future__ import annotations

import io
import re
from contextlib import redirect_stdout, suppress
from typing import TYPE_CHECKING

from fastmcp.server.dependencies import get_context

if TYPE_CHECKING:
    from pyspark.sql import SparkSession


def get_spark_version() -> str:
    spark: SparkSession = get_context().request_context.lifespan_context
    return spark.version


def run_sql_query(query: str) -> str:
    spark: SparkSession = get_context().request_context.lifespan_context
    result = spark.sql(query)
    return result.toPandas().to_json(orient="records")


def get_analyzed_logical_plan_of_query(query: str) -> str:
    # Partialy inspired by the implementation in PySpark-AI
    spark: SparkSession = get_context().request_context.lifespan_context
    df = spark.sql(query)
    with redirect_stdout(io.StringIO()) as stdout_var:
        df.explain(extended=True)

    plan_rows = stdout_var.getvalue().split("\n")
    begin = plan_rows.index("== Analyzed Logical Plan ==")
    end = plan_rows.index("== Optimized Logical Plan ==")

    return "\n".join(plan_rows[begin + 2 : end])


def get_optimized_logical_plan_of_query(query: str) -> str:
    spark: SparkSession = get_context().request_context.lifespan_context
    df = spark.sql(query)
    with redirect_stdout(io.StringIO()) as stdout_var:
        df.explain(extended=True)

    plan_rows = stdout_var.getvalue().split("\n")
    begin = plan_rows.index("== Optimized Logical Plan ==")
    end = plan_rows.index("== Physical Plan ==")

    return "\n".join(plan_rows[begin + 2 : end])


def get_size_in_bytes_estimation_of_query(query: str) -> tuple[float, str]:
    # Partially inpired by https://semyonsinchenko.github.io/ssinchenko/post/estimation-spark-df-size/
    spark: SparkSession = get_context().request_context.lifespan_context
    df = spark.sql(query)
    with redirect_stdout(io.StringIO()) as stdout_var:
        df.explain(mode="cost")

    pattern = r"^.*sizeInBytes=([0-9]+\.[0-9]+)\s(B|KiB|MiB|GiB|TiB|EiB).*$"
    top_line = stdout_var.getvalue().split("\n")[1]
    match = re.match(pattern, top_line)

    if match is not None:
        groups = match.groups()
        return (float(groups[0]), groups[1])
    else:
        return (-1.0, "missing")


def get_tables_from_plan_of_query(query: str) -> list[str]:
    # Inspired by the implementation in PySpark-AI
    analyzed_plan = get_analyzed_logical_plan_of_query(query)
    splits = analyzed_plan.split("\n")
    # For table relations, the table name is the 2nd element in the line
    # It can be one of the followings:
    # 1. "  +- Relation default.foo101"
    # 2. ":        +- Relation default.foo100"
    # 3. "Relation default.foo100"
    tables = []
    for line in splits:
        # if line starts with "Relation" or contains "+- Relation", it is a table relation
        if line.startswith("Relation ") or "+- Relation " in line:
            table_name_with_output = line.split("Relation ", 1)[1].split(" ")[0]
            table_name = table_name_with_output.split("[")[0]
            tables.append(table_name)

    return tables


def get_current_spark_catalog() -> str:
    return get_context().request_context.lifespan_context.catalog.currentCatalog()


def check_database_exists(db_name: str) -> bool:
    return get_context().request_context.lifespan_context.catalog.databaseExists(
        db_name
    )


def get_current_spark_database() -> str:
    return get_context().request_context.lifespan_context.catalog.currentDatabase()


def list_available_databases() -> list[str]:
    spark: SparkSession = get_context().request_context.lifespan_context
    databases = spark.catalog.listDatabases()
    return [str(db) for db in databases]


def list_available_catalogs() -> list[str]:
    spark: SparkSession = get_context().request_context.lifespan_context
    catalogs = spark.catalog.listCatalogs()
    return [str(ct) for ct in catalogs]


def list_available_tables() -> list[str]:
    spark: SparkSession = get_context().request_context.lifespan_context
    tables = spark.catalog.listTables()
    return [str(tb) for tb in tables]


def get_table_comment(table_name: str) -> str:
    # Partially inspired by the implementation in PySpark-AI
    spark: SparkSession = get_context().request_context.lifespan_context
    with suppress(Exception):
        # Get the output of describe table
        outputs = spark.sql("DESC extended " + table_name).collect()
        # Get the table comment from output if the first row is "Comment"
        for row in outputs:
            if row.col_name == "Comment":
                return row.data_type

    # If fail to get table comment, return empty string
    return ""


def get_table_schema(table_name: str) -> str:
    spark: SparkSession = get_context().request_context.lifespan_context
    return spark.table(table_name).schema.json()


def get_output_schema_of_query(query: str) -> str:
    spark: SparkSession = get_context().request_context.lifespan_context
    return spark.sql(query).schema.json()


def read_n_lines_of_text_file(file_path: str, num_lines: int) -> list[str]:
    spark: SparkSession = get_context().request_context.lifespan_context
    rows = spark.read.text(file_path).head(num_lines)
    return [r["value"] for r in rows]
