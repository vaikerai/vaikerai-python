import httpx
import pytest
import respx

from vaikerai.account import Account
from vaikerai.client import Client

router = respx.Router(base_url="https://api.vaikerai.com/v1")
router.route(
    method="GET",
    path="/account",
    name="accounts.current",
).mock(
    return_value=httpx.Response(
        200,
        json={
            "type": "organization",
            "username": "vaikerai",
            "name": "VaikerAI",
            "github_url": "https://github.com/vaikerai",
        },
    )
)
router.route(host="api.vaikerai.com").pass_through()


@pytest.mark.asyncio
@pytest.mark.parametrize("async_flag", [True, False])
async def test_account_current(async_flag):
    client = Client(
        api_token="test-token", transport=httpx.MockTransport(router.handler)
    )

    if async_flag:
        account = await client.accounts.async_current()
    else:
        account = client.accounts.current()

    assert router["accounts.current"].called
    assert isinstance(account, Account)
    assert account.type == "organization"
    assert account.username == "vaikerai"
    assert account.name == "VaikerAI"
    assert account.github_url == "https://github.com/vaikerai"
