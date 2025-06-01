from pyspark_mcp_server import start_mcp_server


if __name__ == "__main__":
    start_mcp_server(host="127.0.0.1", port=8009).run("sse")
