import pytest

import vaikerai


@pytest.mark.vcr("hardware-list.yaml")
@pytest.mark.asyncio
@pytest.mark.parametrize("async_flag", [True, False])
async def test_hardware_list(async_flag):
    if async_flag:
        hardware = await vaikerai.hardware.async_list()
    else:
        hardware = vaikerai.hardware.list()

    assert hardware is not None
    assert isinstance(hardware, list)
    assert len(hardware) > 0
