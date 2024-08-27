import pytest

import vaikerai


@pytest.mark.asyncio
async def test_paginate_with_none_cursor(mock_vaikerai_api_token):
    with pytest.raises(ValueError):
        vaikerai.models.list(None)


@pytest.mark.vcr("collections-list.yaml")
@pytest.mark.asyncio
@pytest.mark.parametrize("async_flag", [True, False])
async def test_paginate(async_flag):
    found = False

    if async_flag:
        async for page in vaikerai.async_paginate(vaikerai.collections.async_list):
            assert page.next is None
            assert page.previous is None

            for collection in page:
                if collection.slug == "text-to-image":
                    found = True
                    break

    else:
        for page in vaikerai.paginate(vaikerai.collections.list):
            assert page.next is None
            assert page.previous is None

            for collection in page:
                if collection.slug == "text-to-image":
                    found = True
                    break

    assert found
