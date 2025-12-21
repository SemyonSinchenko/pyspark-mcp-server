hi# PySpark MCP Server

## Description

PySpark MCP Server is a lightweight server implementation of Model Context Protocol (MCP) for Apache Spark.

The primary purpose of this MCP server is to facilitate query optimization using AI systems. It provides both logical and physical query plans from Spark to AI systems for analysis, along with additional query plan information. Furthermore, the server exposes catalog and table information, enabling data discovery capabilities in data lakes powered by Spark.

## Quick Start

### Installation

```sh
pip install pyspark-mcp
```

### Running the Server

After installation, use the `pyspark-mcp` command to start the server:

```sh
pyspark-mcp --master "local[*]" --host 127.0.0.1 --port 8090
```

The CLI automatically handles `spark-submit` configuration. All standard spark-submit options are supported:

```sh
# With additional Spark configuration
pyspark-mcp --master "local[*]" --conf spark.driver.memory=4g

# YARN cluster mode
pyspark-mcp --master yarn --deploy-mode client --num-executors 4

# With additional JARs
pyspark-mcp --master "local[*]" --jars /path/to/connector.jar

# Preview the spark-submit command without running
pyspark-mcp --master "local[*]" --dry-run

# With GraphFrames package
pyspark-mcp --master "local[*]" --packages io.graphframes:graphframes-spark3_2.12:0.10.0
```

#### CLI Options

| Option | Default | Description |
|--------|---------|-------------|
| `--master` | `local[*]` | Spark master URL |
| `--host` | `127.0.0.1` | MCP server host address |
| `--port` | `8090` | MCP server port number |
| `--spark-submit` | `spark-submit` | Path to spark-submit executable |
| `--dry-run` | - | Print command without executing |

All spark-submit options (`--conf`, `--jars`, `--packages`, `--executor-memory`, etc.) are passed through automatically.

### Adding the running MCP to the Claude-code

```sh
# Must run one server on a different port per Claude instance
claude mcp add --transport http pyspark-mcp http://127.0.0.1:8090/mcp
```

### Dependencies

- Python >=3.11,<4.0
- fastmcp >= 2.10.6
- loguru
- pyspark >= 3.5

### Bundled MCP tools

The following tools are included in the PySpark MCP Server:

| MCP Tool                                        | Description                                                                                   |
|-------------------------------------------------|-----------------------------------------------------------------------------------------------|
| Get the version of PySpark                      | Get the version number from the current PySpark Session                                       |
| Get Analyzed Plan of the query                  | Extracts an analyzed logical plan from the provided SQL query                                 |
| Get Optimized Plan of the query                 | Extracts an optimized logical plan from the provided SQL query                                |
| Get size estimation for the query results       | Extracts a size and units from the query plan explain                                         |
| Get tables from the query plan                  | Extracts all the tables (relations) from the query plan explain                               |
| Get the current Spark Catalog                   | Get the catalog that is the default one for the current SparkSession                          |
| Check does database exist                       | Check if the database with a given name exists in the current Catalog                         |
| Get the current default database                | Get the current default database from the default Catalog                                     |
| List all the databases in the current catalog   | List all the available databases from the current Catalog                                     |
| List available catalogs                         | List all the catalogs available in the current SparkSession                                   |
| List tables in the current catalog              | List all the available tables in the current Spark Catalog                                    |
| Get a comment of the table                      | Extract comment of the table or returns an empty string                                       |
| Get table schema                                | Get the spark schema of the table in the catalog                                              |
| Returns a schema of the result of the SQL query | Run query, get the result, get the schema of the result and return a JSON-value of the schema |
| Read first N lines of the text file             | Read the first N lines of the file as a plain text. Useful to determine the format            |

