# ensures all async tests use the same event loop
# required for pytest-asyncio
pytest_plugins = ("anyio",)