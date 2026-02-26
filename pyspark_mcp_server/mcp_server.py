from __future__ import annotations

import argparse
import atexit
import io
import re
import socket
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
    json_result = cast(pd.DataFrame, result.toPandas()).to_json(orient="records")
    return json_result or "[]"


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
        spark = (
            SparkSession.builder.appName("PySpark MCP")
            .config("spark.network.timeout", "604800s")
            .config("spark.executor.heartbeatInterval", "300s")
            # Disable Spark UI to avoid holding port 4040 after exit
            .config("spark.ui.enabled", "false")
            .getOrCreate()
        )

        # Register atexit handler so spark.stop() runs even on unclean exits
        # (e.g., timeout crashes, SIGKILL). This is a safety net beyond the
        # normal lifespan cleanup below.
        def _stop_spark() -> None:
            try:
                spark.stop()
            except Exception:
                pass

        atexit.register(_stop_spark)

        yield spark
        logger.info("Stopping the SparkSession")
        spark.stop()
        atexit.unregister(_stop_spark)

    mcp = FastMCP(lifespan=lifespan)

    mcp.add_tool(
        Tool.from_function(
            run_sql_query,
            name="run_sql_query",
            description="Run the provided SQL query and return results as JSON",
        ),
    )

    mcp.add_tool(
        Tool.from_function(
            get_spark_version,
            name="get_pyspark_version",
            description="Get the version number from the current PySpark Sessiion",
        ),
    )
    mcp.add_tool(
        Tool.from_function(
            get_analyzed_logical_plan_of_query,
            name="get_analyzed_plan",
            description="Extracts an analyzed logical plan from the provided SQL query",
        ),
    )
    mcp.add_tool(
        Tool.from_function(
            get_optimized_logical_plan_of_query,
            name="get_optimized_plan",
            description="Extracts an optimized logical plan from the provided SQL query",
        ),
    )
    mcp.add_tool(
        Tool.from_function(
            get_size_in_bytes_estimation_of_query,
            name="get_size_estimation",
            description="Extracts a size and units from the query plan explain",
        ),
    )
    mcp.add_tool(
        Tool.from_function(
            get_tables_from_plan_of_query,
            name="get_tables_from_plan",
            description="Extracts all the tables (relations) from the query plan explain",
        ),
    )
    mcp.add_tool(
        Tool.from_function(
            get_current_spark_catalog,
            name="get_current_catalog",
            description="Get the catalog that is the default one for the current SparkSession",
        ),
    )
    mcp.add_tool(
        Tool.from_function(
            check_database_exists,
            name="check_database_exists",
            description="Check if the database with a given name exists in the current Catalog",
        ),
    )
    mcp.add_tool(
        Tool.from_function(
            get_current_spark_database,
            name="get_current_database",
            description="Get the current default database from the default Catalog",
        ),
    )
    mcp.add_tool(
        Tool.from_function(
            list_available_databases,
            name="list_databases",
            description="List all the available databases from the current Catalog",
        ),
    )
    mcp.add_tool(
        Tool.from_function(
            list_available_catalogs,
            name="list_catalogs",
            description="List all the catalogs available in the current SparkSession",
        ),
    )
    mcp.add_tool(
        Tool.from_function(
            list_available_tables,
            name="list_tables",
            description="List all the available tables in the current Spark Catalog",
        ),
    )
    mcp.add_tool(
        Tool.from_function(
            get_table_comment,
            name="get_table_comment",
            description="Extract comment of the table or returns an empty string",
        ),
    )
    mcp.add_tool(
        Tool.from_function(
            get_table_schema,
            name="get_table_schema",
            description="Get the spark schema of the table in the catalog",
        ),
    )
    mcp.add_tool(
        Tool.from_function(
            get_output_schema_of_query,
            name="get_query_schema",
            description="Run query, get the result, get the schema of the result "
            "and return a JSON-value of the schema",
        ),
    )
    mcp.add_tool(
        Tool.from_function(
            read_n_lines_of_text_file,
            name="read_text_file_lines",
            description="Read the first N lines of the file as a plain text. "
            "Useful to determine the format",
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

    # Check port availability before starting SparkSession (which takes seconds)
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind((args.host, args.port))
    except OSError:
        logger.error(
            f"Port {args.port} is already in use on {args.host}. "
            f"Use --port to specify a different port."
        )
        sys.exit(1)

    # Monkey-patch socket creation to set SO_REUSEADDR and SO_REUSEPORT on all
    # TCP sockets. This allows immediate port reuse after exit, even when
    # connections are stuck in TIME_WAIT (common after timeout crashes).
    # SO_REUSEPORT is critical on macOS where SO_REUSEADDR alone is insufficient.
    original_socket = socket.socket

    def patched_socket(
        family: int = socket.AF_INET,
        type: int = socket.SOCK_STREAM,
        *args: Any,
        **kwargs: Any,
    ) -> socket.socket:
        sock = original_socket(family, type, *args, **kwargs)
        if family in (socket.AF_INET, socket.AF_INET6) and type == socket.SOCK_STREAM:
            try:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            except OSError:
                pass
            if hasattr(socket, "SO_REUSEPORT"):
                try:
                    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
                except OSError:
                    pass
        return sock

    socket.socket = patched_socket  # type: ignore[misc]

    goodbye = """

    *    .  *       .             *
         *       *        .   *     .  *
   .  *     Spark stopped.     .
        .   Session ended cleanly.  *
     *        Goodbye!       .     *
  .      *        .    *   .         .
    """

    try:
        start_mcp_server().run(transport="http", port=args.port, host=args.host)
    except (KeyboardInterrupt, SystemExit):
        print(goodbye)
    finally:
        socket.socket = original_socket


if __name__ == "__main__":
    main()
