from contextlib import asynccontextmanager
from typing import AsyncIterator

import loguru
from fastmcp import FastMCP
from pyspark.sql import SparkSession

from . import tools

logger = loguru.logger


def start_mcp_server(host: str, port: int) -> FastMCP:
    """Start MCP server.

    It is assumed that the SparkSession already exists.
    Use spark-submit and a wrapper to run it.

    Args:
        host (str): hostname of the MCP Server
        port (str): port of the MCP Server
    """

    # Context is inspired by the implementation in the LakeSail
    @asynccontextmanager
    async def lifespan(server: FastMCP) -> AsyncIterator[SparkSession]:
        logger.info("Starting the SparkSession")
        spark = SparkSession.builder.appName("PySpark MCP").getOrCreate()
        yield spark
        logger.info("Stopping the SparkSession")
        spark.stop()

    mcp = FastMCP(host=host, port=port, lifespan=lifespan)

    mcp.add_tool(
        tools.run_sql_query,
        name="Run SQL quey",
        description="Run the provided SQL query and return results as JSON",
    )
    mcp.add_tool(
        tools.get_spark_version,
        name="Get the version of PySpark",
        description="Get the version number from the current PySpark Sessiion",
    )
    mcp.add_tool(
        tools.get_analyzed_logical_plan_of_query,
        name="Get Analyzed Plan of the query",
        description="Extracts an analyzed logical plan from the provided SQL query",
    )
    mcp.add_tool(
        tools.get_optimized_logical_plan_of_query,
        name="Get Optimized Plan of the query",
        description="Extracts an optimized logical plan from the provided SQL query",
    )
    mcp.add_tool(
        tools.get_size_in_bytes_estimation_of_query,
        name="Get size estimation for the query results",
        description="Extracts a size and units from the query plan explain",
    )
    mcp.add_tool(
        tools.get_tables_from_plan_of_query,
        name="Get tables from the query plan",
        description="Extracts all the tables (relations) from the query plan explain",
    )
    mcp.add_tool(
        tools.get_current_spark_catalog,
        name="Get the current Spark Catalog",
        description="Get the catalog that is the default one for the current SparkSession",
    )
    mcp.add_tool(
        tools.check_database_exists,
        name="Check does database exist",
        description="Check if the database with a given name exists in the current Catalog",
    )
    mcp.add_tool(
        tools.get_current_spark_database,
        name="Get the current default database",
        description="Get the current default database from the default Catalog",
    )
    mcp.add_tool(
        tools.list_available_databases,
        name="List all the databases in the current catalog",
        description="List all the available databases from the current Catalog",
    )
    mcp.add_tool(
        tools.list_available_catalogs,
        name="List available catalogs",
        description="List all the catalogs available in the current SparkSession",
    )
    mcp.add_tool(
        tools.list_available_tables,
        name="List tables in the current catalog",
        description="List all the available tables in the current Spark Catalog",
    )
    mcp.add_tool(
        tools.get_table_comment,
        name="Get a comment of the table",
        description="Extract comment of the table or returns an empty string",
    )
    mcp.add_tool(
        tools.get_table_schema,
        name="Get table schema",
        description="Get the spark schema of the table in the catalog",
    )
    mcp.add_tool(
        tools.get_output_schema_of_query,
        name="Returns a schema of the result of the SQL query",
        description="Run query, get the result, get the schema of the result and return a JSON-value of the schema",
    )
    mcp.add_tool(
        tools.read_n_lines_of_text_file,
        name="Read first N lines of the text file",
        description="Read the first N lines of the file as a plain text. Useful to determine the format",
    )

    return mcp
