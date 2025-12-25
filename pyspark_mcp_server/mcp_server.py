from __future__ import annotations

import argparse
import io
import re
import signal
import sys
from contextlib import asynccontextmanager, redirect_stdout, suppress
from typing import Any, AsyncIterator, cast

import loguru
import pandas as pd
from fastmcp import FastMCP
from fastmcp.server.dependencies import get_context
from fastmcp.tools import Tool
from pyspark.sql import SparkSession

logger = loguru.logger


def get_spark_version() -> str:
    spark: SparkSession = get_context().request_context.lifespan_context  # type: ignore
    return spark.version


def run_sql_query(query: str) -> str:
    spark: SparkSession = get_context().request_context.lifespan_context  # type: ignore
    result = spark.sql(query)
    return cast(pd.DataFrame, result.toPandas()).to_json(orient="records")


def get_analyzed_logical_plan_of_query(query: str) -> str:
    # Partialy inspired by the implementation in PySpark-AI
    spark: SparkSession = get_context().request_context.lifespan_context  # type: ignore
    df = spark.sql(query)
    with redirect_stdout(io.StringIO()) as stdout_var:
        df.explain(extended=True)

    plan_rows = stdout_var.getvalue().split("\n")
    begin = plan_rows.index("== Analyzed Logical Plan ==")
    end = plan_rows.index("== Optimized Logical Plan ==")

    return "\n".join(plan_rows[begin + 2 : end])


def get_optimized_logical_plan_of_query(query: str) -> str:
    spark: SparkSession = get_context().request_context.lifespan_context  # type: ignore
    df = spark.sql(query)
    with redirect_stdout(io.StringIO()) as stdout_var:
        df.explain(extended=True)

    plan_rows = stdout_var.getvalue().split("\n")
    begin = plan_rows.index("== Optimized Logical Plan ==")
    end = plan_rows.index("== Physical Plan ==")

    return "\n".join(plan_rows[begin + 2 : end])


def get_size_in_bytes_estimation_of_query(query: str) -> tuple[float, str]:
    # Partially inpired by https://semyonsinchenko.github.io/ssinchenko/post/estimation-spark-df-size/
    spark: SparkSession = get_context().request_context.lifespan_context  # type: ignore
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


def get_current_spark_catalog() -> str | Any:
    return get_context().request_context.lifespan_context.catalog.currentCatalog()  # type: ignore


def check_database_exists(db_name: str) -> bool | Any:
    return get_context().request_context.lifespan_context.catalog.databaseExists(db_name)  # type: ignore


def get_current_spark_database() -> str | Any:
    return get_context().request_context.lifespan_context.catalog.currentDatabase()  # type: ignore


def list_available_databases() -> list[str]:
    spark: SparkSession = get_context().request_context.lifespan_context  # type: ignore
    databases = spark.catalog.listDatabases()
    return [str(db) for db in databases]


def list_available_catalogs() -> list[str]:
    spark: SparkSession = get_context().request_context.lifespan_context  # type: ignore
    catalogs = spark.catalog.listCatalogs()
    return [str(ct) for ct in catalogs]


def list_available_tables() -> list[str]:
    spark: SparkSession = get_context().request_context.lifespan_context  # type: ignore
    tables = spark.catalog.listTables()
    return [str(tb) for tb in tables]


def get_table_comment(table_name: str) -> str | Any:
    # Partially inspired by the implementation in PySpark-AI
    spark: SparkSession = get_context().request_context.lifespan_context  # type: ignore
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
    spark: SparkSession = get_context().request_context.lifespan_context  # type: ignore
    return spark.table(table_name).schema.json()


def get_output_schema_of_query(query: str) -> str:
    spark: SparkSession = get_context().request_context.lifespan_context  # type: ignore
    return spark.sql(query).schema.json()


def read_n_lines_of_text_file(file_path: str, num_lines: int) -> list[str]:
    spark: SparkSession = get_context().request_context.lifespan_context  # type: ignore
    rows = spark.read.text(file_path).head(num_lines)
    return [r["value"] for r in rows]


def start_mcp_server() -> FastMCP:
    """Start MCP server.

    It is assumed that the SparkSession already exists.
    Use spark-submit and a wrapper to run it.
    """

    # Context is inspired by the implementation in the LakeSail
    @asynccontextmanager
    async def lifespan(server: FastMCP) -> AsyncIterator[SparkSession]:
        logger.info("Starting the SparkSession")
        spark = SparkSession.builder.appName("PySpark MCP").getOrCreate()
        yield spark
        logger.info("Stopping the SparkSession")
        spark.stop()

    mcp = FastMCP(lifespan=lifespan)

    mcp.add_tool(
        Tool.from_function(
            run_sql_query,
            name="Run SQL query",
            description="Run the provided SQL query and return results as JSON",
        ),
    )

    mcp.add_tool(
        Tool.from_function(
            get_spark_version,
            name="Get the version of PySpark",
            description="Get the version number from the current PySpark Sessiion",
        ),
    )
    mcp.add_tool(
        Tool.from_function(
            get_analyzed_logical_plan_of_query,
            name="Get Analyzed Plan of the query",
            description="Extracts an analyzed logical plan from the provided SQL query",
        ),
    )
    mcp.add_tool(
        Tool.from_function(
            get_optimized_logical_plan_of_query,
            name="Get Optimized Plan of the query",
            description="Extracts an optimized logical plan from the provided SQL query",
        ),
    )
    mcp.add_tool(
        Tool.from_function(
            get_size_in_bytes_estimation_of_query,
            name="Get size estimation for the query results",
            description="Extracts a size and units from the query plan explain",
        ),
    )
    mcp.add_tool(
        Tool.from_function(
            get_tables_from_plan_of_query,
            name="Get tables from the query plan",
            description="Extracts all the tables (relations) from the query plan explain",
        ),
    )
    mcp.add_tool(
        Tool.from_function(
            get_current_spark_catalog,
            name="Get the current Spark Catalog",
            description="Get the catalog that is the default one for the current SparkSession",
        ),
    )
    mcp.add_tool(
        Tool.from_function(
            check_database_exists,
            name="Check does database exist",
            description="Check if the database with a given name exists in the current Catalog",
        ),
    )
    mcp.add_tool(
        Tool.from_function(
            get_current_spark_database,
            name="Get the current default database",
            description="Get the current default database from the default Catalog",
        ),
    )
    mcp.add_tool(
        Tool.from_function(
            list_available_databases,
            name="List all the databases in the current catalog",
            description="List all the available databases from the current Catalog",
        ),
    )
    mcp.add_tool(
        Tool.from_function(
            list_available_catalogs,
            name="List available catalogs",
            description="List all the catalogs available in the current SparkSession",
        ),
    )
    mcp.add_tool(
        Tool.from_function(
            list_available_tables,
            name="List tables in the current catalog",
            description="List all the available tables in the current Spark Catalog",
        ),
    )
    mcp.add_tool(
        Tool.from_function(
            get_table_comment,
            name="Get a comment of the table",
            description="Extract comment of the table or returns an empty string",
        ),
    )
    mcp.add_tool(
        Tool.from_function(
            get_table_schema,
            name="Get table schema",
            description="Get the spark schema of the table in the catalog",
        ),
    )
    mcp.add_tool(
        Tool.from_function(
            get_output_schema_of_query,
            name="Returns a schema of the result of the SQL query",
            description="Run query, get the result, get the schema of the result and return a JSON-value of the schema",
        ),
    )
    mcp.add_tool(
        Tool.from_function(
            read_n_lines_of_text_file,
            name="Read first N lines of the text file",
            description="Read the first N lines of the file as a plain text. Useful to determine the format",
        ),
    )

    return mcp


def main() -> None:
    parser = argparse.ArgumentParser(description="Start MCP server")
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host address (default: 127.0.0.1)",
    )
    parser.add_argument("--port", type=int, default=8090, help="Port number (default: 8090)")

    args = parser.parse_args()

    # Set up signal handlers for clean shutdown
    # This ensures the server stops properly when receiving SIGINT (CTRL-C) or SIGTERM,
    # preventing port binding issues on restart
    def signal_handler(signum: int, frame: Any) -> None:
        logger.info(f"Received signal {signum}, shutting down gracefully...")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        start_mcp_server().run(transport="http", port=args.port, host=args.host)
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received, shutting down...")


if __name__ == "__main__":
    main()
