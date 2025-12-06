"""CLI wrapper for pyspark-mcp-server.

This module provides a smart CLI that handles both Spark and MCP arguments,
automatically locating the mcp_server.py script and invoking spark-submit.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import click

# Common spark-submit options that take a value
SPARK_OPTIONS_WITH_VALUE = {
    "--master",
    "--deploy-mode",
    "--class",
    "--name",
    "--jars",
    "--packages",
    "--exclude-packages",
    "--repositories",
    "--py-files",
    "--files",
    "--conf",
    "--properties-file",
    "--driver-memory",
    "--driver-java-options",
    "--driver-library-path",
    "--driver-class-path",
    "--executor-memory",
    "--proxy-user",
    "--driver-cores",
    "--total-executor-cores",
    "--executor-cores",
    "--num-executors",
    "--principal",
    "--keytab",
    "--queue",
    "--archives",
}

# Spark-submit boolean flags (no value)
SPARK_FLAGS = {
    "--verbose",
    "--supervise",
    "--help",
    "--version",
}


def get_mcp_server_path() -> Path:
    """Get the path to mcp_server.py in the installed package."""
    return Path(__file__).parent / "mcp_server.py"


def parse_spark_and_mcp_args(args: tuple[str, ...]) -> tuple[list[str], list[str]]:
    """Separate spark-submit arguments from MCP server arguments.

    Args:
        args: All command line arguments

    Returns:
        Tuple of (spark_args, mcp_args)
    """
    spark_args: list[str] = []
    mcp_args: list[str] = []
    args_list = list(args)
    i = 0

    while i < len(args_list):
        arg = args_list[i]

        # Check if it's a spark option with value
        if arg in SPARK_OPTIONS_WITH_VALUE:
            spark_args.append(arg)
            if i + 1 < len(args_list):
                i += 1
                spark_args.append(args_list[i])
            else:
                # Missing required value for spark option
                raise click.UsageError(f"Option {arg} requires a value")
        # Check for --conf style with = (e.g., --conf spark.executor.memory=4g)
        elif arg.startswith("--conf="):
            spark_args.append(arg)
        # Check for other spark options with = (e.g., --master=local[*], --jars=foo.jar)
        elif any(arg.startswith(f"{opt}=") for opt in SPARK_OPTIONS_WITH_VALUE):
            spark_args.append(arg)
        # Check if it's a spark flag
        elif arg in SPARK_FLAGS:
            spark_args.append(arg)
        # Everything else goes to MCP
        else:
            mcp_args.append(arg)

        i += 1

    return spark_args, mcp_args


@click.command(context_settings={"ignore_unknown_options": True, "allow_extra_args": True})
@click.option(
    "--master",
    default="local[*]",
    help="Spark master URL (default: local[*])",
)
@click.option(
    "--host",
    default="127.0.0.1",
    help="MCP server host address (default: 127.0.0.1)",
)
@click.option(
    "--port",
    default=8090,
    type=click.IntRange(1, 65535),
    help="MCP server port number (default: 8090)",
)
@click.option(
    "--spark-submit",
    "spark_submit_path",
    default="spark-submit",
    help="Path to spark-submit executable (default: spark-submit)",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Print the spark-submit command without executing it",
)
@click.pass_context
def main(  # noqa: C901
    ctx: click.Context,
    master: str,
    host: str,
    port: int,
    spark_submit_path: str,
    dry_run: bool,
) -> None:
    """Start the PySpark MCP server via spark-submit.

    This command wraps spark-submit to launch the MCP server with proper
    Spark configuration. All standard spark-submit options are supported.

    Examples:

        # Basic local mode
        pyspark-mcp --master "local[4]" --host 0.0.0.0 --port 8090

        # With additional Spark configuration
        pyspark-mcp --master "local[*]" --conf spark.driver.memory=4g

        # YARN cluster mode
        pyspark-mcp --master yarn --deploy-mode client --num-executors 4

        # With additional JARs
        pyspark-mcp --master "local[*]" --jars /path/to/connector.jar
    """
    # Get path to mcp_server.py
    mcp_server_path = get_mcp_server_path()

    if not mcp_server_path.exists():
        click.echo(f"Error: Could not find mcp_server.py at {mcp_server_path}", err=True)
        sys.exit(1)

    # Parse extra args to separate Spark args from any additional MCP args
    spark_args, extra_mcp_args = parse_spark_and_mcp_args(tuple(ctx.args))

    # Build the spark-submit command
    cmd = [
        spark_submit_path,
        "--master",
        master,
    ]

    # Add any additional spark args
    # Remove --master and its value from spark_args if --master was provided via Click option
    filtered_spark_args = []
    skip_next = False
    for i, arg in enumerate(spark_args):
        if skip_next:
            skip_next = False
            continue
        if arg == "--master":
            # Skip this and the next value
            skip_next = True
            continue
        filtered_spark_args.append(arg)
    cmd.extend(filtered_spark_args)

    # Add the script path
    cmd.append(str(mcp_server_path))

    # Add MCP server arguments
    cmd.extend(["--host", host, "--port", str(port)])

    # Add any extra MCP args, filtering out --host and --port (and their values)
    def filter_host_port(args: list[str]) -> list[str]:
        filtered = []
        skip_next = False
        for i, arg in enumerate(args):
            if skip_next:
                skip_next = False
                continue
            if arg in ("--host", "--port"):
                skip_next = True
                continue
            if arg.startswith("--host=") or arg.startswith("--port="):
                continue
            filtered.append(arg)
        return filtered

    cmd.extend(filter_host_port(extra_mcp_args))

    if dry_run:
        click.echo(" ".join(cmd))
        return

    # Set up environment - ensure PYTHONPATH includes the package
    env = os.environ.copy()
    package_dir = str(Path(mcp_server_path).parent)
    env["PYTHONPATH"] = package_dir + os.pathsep + env.get("PYTHONPATH", "")

    # Execute spark-submit
    try:
        result = subprocess.run(cmd, env=env)
        sys.exit(result.returncode)
    except FileNotFoundError:
        click.echo(
            f"Error: Could not find '{spark_submit_path}'. "
            "Make sure Spark is installed and spark-submit is in your PATH, "
            "or use --spark-submit to specify the path.",
            err=True,
        )
        sys.exit(1)
    except KeyboardInterrupt:
        click.echo("\nShutting down...")
        sys.exit(0)


if __name__ == "__main__":
    main()
