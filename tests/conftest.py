import os
from collections.abc import Iterator

import pytest
from sqlalchemy import create_engine

from genai_lens.server.db import Base


@pytest.fixture
def db_url() -> Iterator[str]:
    """SQLite in memory by default; set GENAI_LENS_TEST_DATABASE_URL to run against Postgres."""
    url = os.getenv("GENAI_LENS_TEST_DATABASE_URL", "sqlite://")
    yield url
    if not url.startswith("sqlite"):
        engine = create_engine(url)
        Base.metadata.drop_all(engine)
        engine.dispose()
