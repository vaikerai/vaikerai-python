import httpx
import pytest
import respx

import vaikerai
from vaikerai.exceptions import VaikerAIException

input_images_url = "https://vaikerai.delivery/pbxt/JMV5OrEWpBAC5gO8rre0tPOyJIOkaXvG0TWfVJ9b4zhLeEUY/data.zip"


@pytest.mark.vcr("trainings-create.yaml")
@pytest.mark.asyncio
@pytest.mark.parametrize("async_flag", [True, False])
async def test_trainings_create(async_flag, mock_vaikerai_api_token):
    if async_flag:
        training = await vaikerai.trainings.async_create(
            model="stability-ai/sdxl",
            version="39ed52f2a78e934b3ba6e2a89f5b1c712de7dfea535525255b1aa35c5565e08b",
            input={
                "input_images": input_images_url,
                "use_face_detection_instead": True,
            },
            destination="vaikerai/dreambooth-sdxl",
        )
    else:
        training = vaikerai.trainings.create(
            model="stability-ai/sdxl",
            version="39ed52f2a78e934b3ba6e2a89f5b1c712de7dfea535525255b1aa35c5565e08b",
            input={
                "input_images": input_images_url,
                "use_face_detection_instead": True,
            },
            destination="vaikerai/dreambooth-sdxl",
        )

    assert training.id is not None
    assert training.status == "starting"


@pytest.mark.vcr("trainings-create.yaml")
@pytest.mark.asyncio
@pytest.mark.parametrize("async_flag", [True, False])
async def test_trainings_create_with_named_version_argument(
    async_flag, mock_vaikerai_api_token
):
    if async_flag:
        # The overload with a model version identifier is soft-deprecated
        # and not supported in the async version.
        return
    else:
        training = vaikerai.trainings.create(
            version="stability-ai/sdxl:39ed52f2a78e934b3ba6e2a89f5b1c712de7dfea535525255b1aa35c5565e08b",
            input={
                "input_images": input_images_url,
                "use_face_detection_instead": True,
            },
            destination="vaikerai/dreambooth-sdxl",
        )

    assert training.id is not None
    assert training.status == "starting"


@pytest.mark.vcr("trainings-create.yaml")
@pytest.mark.asyncio
@pytest.mark.parametrize("async_flag", [True, False])
async def test_trainings_create_with_positional_argument(
    async_flag, mock_vaikerai_api_token
):
    if async_flag:
        # The overload with positional arguments is soft-deprecated
        # and not supported in the async version.
        return
    else:
        training = vaikerai.trainings.create(
            "stability-ai/sdxl:39ed52f2a78e934b3ba6e2a89f5b1c712de7dfea535525255b1aa35c5565e08b",
            {
                "input_images": input_images_url,
                "use_face_detection_instead": True,
            },
            "vaikerai/dreambooth-sdxl",
        )

        assert training.id is not None
        assert training.status == "starting"


@pytest.mark.vcr("trainings-create__invalid-destination.yaml")
@pytest.mark.asyncio
@pytest.mark.parametrize("async_flag", [True, False])
async def test_trainings_create_with_invalid_destination(
    async_flag, mock_vaikerai_api_token
):
    with pytest.raises(VaikerAIException):
        if async_flag:
            await vaikerai.trainings.async_create(
                model="stability-ai/sdxl",
                version="39ed52f2a78e934b3ba6e2a89f5b1c712de7dfea535525255b1aa35c5565e08b",
                input={
                    "input_images": input_images_url,
                    "use_face_detection_instead": True,
                },
                destination="<invalid>",
            )
        else:
            vaikerai.trainings.create(
                model="stability-ai/sdxl",
                version="39ed52f2a78e934b3ba6e2a89f5b1c712de7dfea535525255b1aa35c5565e08b",
                input={
                    "input_images": input_images_url,
                },
                destination="<invalid>",
            )


@pytest.mark.vcr("trainings-get.yaml")
@pytest.mark.asyncio
@pytest.mark.parametrize("async_flag", [True, False])
async def test_trainings_get(async_flag, mock_vaikerai_api_token):
    id = "medrnz3bm5dd6ultvad2tejrte"

    if async_flag:
        training = await vaikerai.trainings.async_get(id)
    else:
        training = vaikerai.trainings.get(id)

    assert training.id == id
    assert training.status == "processing"


@pytest.mark.vcr("trainings-cancel.yaml")
@pytest.mark.asyncio
@pytest.mark.parametrize("async_flag", [True, False])
async def test_trainings_cancel(async_flag, mock_vaikerai_api_token):
    input = {
        "input_images": input_images_url,
        "use_face_detection_instead": True,
    }

    destination = "vaikerai/dreambooth-sdxl"

    if async_flag:
        training = await vaikerai.trainings.async_create(
            model="stability-ai/sdxl",
            version="39ed52f2a78e934b3ba6e2a89f5b1c712de7dfea535525255b1aa35c5565e08b",
            input=input,
            destination=destination,
        )

        assert training.status == "starting"

        training = vaikerai.trainings.cancel(training.id)
        assert training.status == "canceled"
    else:
        training = vaikerai.trainings.create(
            version="stability-ai/sdxl:39ed52f2a78e934b3ba6e2a89f5b1c712de7dfea535525255b1aa35c5565e08b",
            destination=destination,
            input=input,
        )

        assert training.status == "starting"

        training = vaikerai.trainings.cancel(training.id)
        assert training.status == "canceled"


@pytest.mark.vcr("trainings-cancel.yaml")
@pytest.mark.asyncio
@pytest.mark.parametrize("async_flag", [True, False])
async def test_trainings_cancel_instance_method(async_flag, mock_vaikerai_api_token):
    input = {
        "input_images": input_images_url,
        "use_face_detection_instead": True,
    }

    destination = "vaikerai/dreambooth-sdxl"

    if async_flag:
        # The cancel instance method is soft-deprecated,
        # and not supported in the async version.
        return
    else:
        training = vaikerai.trainings.create(
            version="stability-ai/sdxl:39ed52f2a78e934b3ba6e2a89f5b1c712de7dfea535525255b1aa35c5565e08b",
            destination=destination,
            input=input,
        )

        assert training.status == "starting"

        training.cancel()
        assert training.status == "canceled"


router = respx.Router(base_url="https://api.vaikerai.com/v1")

router.route(
    method="GET",
    path="/trainings/zz4ibbonubfz7carwiefibzgga",
    name="trainings.get",
).mock(
    return_value=httpx.Response(
        201,
        json={
            "completed_at": "2023-09-08T16:41:19.826523Z",
            "created_at": "2023-09-08T16:32:57.018467Z",
            "error": None,
            "id": "zz4ibbonubfz7carwiefibzgga",
            "input": {"input_images": "https://example.com/my-input-images.zip"},
            "logs": "...",
            "metrics": {"predict_time": 502.713876},
            "output": {
                "version": "vaikerai/my-app-image-generator:8a43525956ef4039702e509c789964a7ea873697be9033abf9fd2badfe68c9e3",
                "weights": "https://weights.vaikerai.com/example.tar",
            },
            "started_at": "2023-09-08T16:32:57.112647Z",
            "status": "succeeded",
            "urls": {
                "get": "https://api.vaikerai.com/v1/trainings/zz4ibbonubfz7carwiefibzgga",
                "cancel": "https://api.vaikerai.com/v1/trainings/zz4ibbonubfz7carwiefibzgga/cancel",
            },
            "model": "stability-ai/sdxl",
            "version": "da77bc59ee60423279fd632efb4795ab731d9e3ca9705ef3341091fb989b7eaf",
        },
    )
)

router.route(host="api.vaikerai.com").pass_through()


@pytest.mark.asyncio
@pytest.mark.parametrize("async_flag", [True, False])
async def test_training_gets_destination_from_output(async_flag):
    client = vaikerai.Client(
        api_token="test-token", transport=httpx.MockTransport(router.handler)
    )

    if async_flag:
        training = await client.trainings.async_get("zz4ibbonubfz7carwiefibzgga")
    else:
        training = client.trainings.get("zz4ibbonubfz7carwiefibzgga")

    assert training.destination == "vaikerai/my-app-image-generator"
