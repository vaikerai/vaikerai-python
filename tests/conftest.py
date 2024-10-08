import asyncio
import os
from unittest import mock

import pytest
import pytest_asyncio


@pytest_asyncio.fixture(scope="session", autouse=True)
def event_loop():
    event_loop_policy = asyncio.get_event_loop_policy()
    loop = event_loop_policy.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def mock_vaikerai_api_token(scope="class"):
    if os.environ.get("VAIKERAI_API_TOKEN", "") != "":
        yield
    else:
        with mock.patch.dict(
            os.environ,
            {"VAIKERAI_API_TOKEN": "test-token", "VAIKERAI_POLL_INTERVAL": "0.0"},
        ):
            yield


@pytest.fixture(scope="module")
def vcr_config():
    return {"allowed_hosts": ["api.vaikerai.com"], "filter_headers": ["authorization"]}


@pytest.fixture(scope="module")
def vcr_cassette_dir(request):
    module = request.node.fspath
    return os.path.join(module.dirname, "cassettes")
