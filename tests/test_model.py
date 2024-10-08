import pytest

import vaikerai
from vaikerai.model import Model, Page


@pytest.mark.vcr("models-get.yaml")
@pytest.mark.asyncio
@pytest.mark.parametrize("async_flag", [True, False])
async def test_models_get(async_flag):
    if async_flag:
        sdxl = await vaikerai.models.async_get("stability-ai/sdxl")
    else:
        sdxl = vaikerai.models.get("stability-ai/sdxl")

    assert sdxl is not None
    assert sdxl.owner == "stability-ai"
    assert sdxl.name == "sdxl"
    assert sdxl.visibility == "public"

    if async_flag:
        empty = await vaikerai.models.async_get("mattt/empty")
    else:
        empty = vaikerai.models.get("mattt/empty")

    assert empty.default_example is None


@pytest.mark.vcr("models-list.yaml")
@pytest.mark.asyncio
@pytest.mark.parametrize("async_flag", [True, False])
async def test_models_list(async_flag):
    if async_flag:
        models = await vaikerai.models.async_list()
    else:
        models = vaikerai.models.list()

    assert len(models) > 0
    assert models[0].owner is not None
    assert models[0].name is not None
    assert models[0].visibility == "public"


@pytest.mark.vcr("models-list__pagination.yaml")
@pytest.mark.asyncio
@pytest.mark.parametrize("async_flag", [True, False])
async def test_models_list_pagination(async_flag):
    if async_flag:
        page1 = await vaikerai.models.async_list()
    else:
        page1 = vaikerai.models.list()
    assert len(page1) > 0
    assert page1.next is not None

    if async_flag:
        page2 = await vaikerai.models.async_list(cursor=page1.next)
    else:
        page2 = vaikerai.models.list(cursor=page1.next)
    assert len(page2) > 0
    assert page2.previous is not None


@pytest.mark.vcr("models-create.yaml")
@pytest.mark.asyncio
@pytest.mark.parametrize("async_flag", [True, False])
async def test_models_create(async_flag):
    if async_flag:
        model = await vaikerai.models.async_create(
            owner="test",
            name="python-example",
            visibility="private",
            hardware="cpu",
            description="An example model",
        )
    else:
        model = vaikerai.models.create(
            owner="test",
            name="python-example",
            visibility="private",
            hardware="cpu",
            description="An example model",
        )

    assert model.owner == "test"
    assert model.name == "python-example"
    assert model.visibility == "private"


@pytest.mark.vcr("models-create.yaml")
@pytest.mark.asyncio
@pytest.mark.parametrize("async_flag", [True, False])
async def test_models_create_with_positional_arguments(async_flag):
    if async_flag:
        model = await vaikerai.models.async_create(
            "test",
            "python-example",
            visibility="private",
            hardware="cpu",
        )
    else:
        model = vaikerai.models.create(
            "test",
            "python-example",
            visibility="private",
            hardware="cpu",
        )

    assert model.owner == "test"
    assert model.name == "python-example"
    assert model.visibility == "private"


@pytest.mark.vcr("models-predictions-create.yaml")
@pytest.mark.asyncio
@pytest.mark.parametrize("async_flag", [True, False])
async def test_models_predictions_create(async_flag):
    input = {
        "prompt": "Please write a haiku about llamas.",
    }

    if async_flag:
        prediction = await vaikerai.models.predictions.async_create(
            "meta/llama-2-70b-chat", input=input
        )
    else:
        prediction = vaikerai.models.predictions.create(
            "meta/llama-2-70b-chat", input=input
        )

    assert prediction.id is not None
    # assert prediction.model == "meta/llama-2-70b-chat"
    assert prediction.model == "vaikerai/lifeboat-70b"  # FIXME: this is temporary
    assert prediction.status == "starting"


@pytest.mark.vcr("models-search.yaml")
@pytest.mark.asyncio
@pytest.mark.parametrize("async_flag", [True, False])
async def test_models_search(async_flag):
    query = "llama"

    if async_flag:
        page = await vaikerai.models.async_search(query)
    else:
        page = vaikerai.models.search(query)

    assert isinstance(page, Page)
    assert len(page.results) > 0

    for model in page.results:
        assert isinstance(model, Model)
        assert model.id is not None
        assert model.owner is not None
        assert model.name is not None

    assert any("meta" in model.name.lower() for model in page.results)
