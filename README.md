# VaikerAI Python client

This is a Python client for [VaikerAI](https://vaikerai.com). It lets you run models from your Python code or Jupyter notebook, and do various other things on VaikerAI.

## Requirements

- Python 3.8+

## Install

```sh
pip install vaikerai
```

## Authenticate

Before running any Python scripts that use the API, you need to set your VaikerAI API token in your environment.

Grab your token from [vaikerai.com/account](https://vaikerai.com/account) and set it as an environment variable:

```
export VAIKERAI_API_TOKEN=<your token>
```

We recommend not adding the token directly to your source code, because you don't want to put your credentials in source control. If anyone used your API key, their usage would be charged to your account.

## Run a model

Create a new Python file and add the following code, 
replacing the model identifier and input with your own:

```python
>>> import vaikerai
>>> vaikerai.run(
        "stability-ai/stable-diffusion:27b93a2413e7f36cd83da926f3656280b2931564ff050bf9575f1fdf9bcd7478",
        input={"prompt": "a 19th century portrait of a wombat gentleman"}
    )

['https://vaikerai.com/api/models/stability-ai/stable-diffusion/files/50fcac81-865d-499e-81ac-49de0cb79264/out-0.png']
```

> [!TIP]
> You can also use the VaikerAI client asynchronously by prepending `async_` to the method name. 
> 
> Here's an example of how to run several predictions concurrently and wait for them all to complete:
>
> ```python
> import asyncio
> import vaikerai
> 
> # https://vaikerai.com/stability-ai/sdxl
> model_version = "stability-ai/sdxl:39ed52f2a78e934b3ba6e2a89f5b1c712de7dfea535525255b1aa35c5565e08b"
> prompts = [
>     f"A chariot pulled by a team of {count} rainbow unicorns"
>     for count in ["two", "four", "six", "eight"]
> ]
>
> async with asyncio.TaskGroup() as tg:
>     tasks = [
>         tg.create_task(vaikerai.async_run(model_version, input={"prompt": prompt}))
>         for prompt in prompts
>     ]
>
> results = await asyncio.gather(*tasks)
> print(results)
> ```

To run a model that takes a file input you can pass either
a URL to a publicly accessible file on the Internet
or a handle to a file on your local device.

```python
>>> output = vaikerai.run(
        "andreasjansson/blip-2:f677695e5e89f8b236e52ecd1d3f01beb44c34606419bcc19345e046d8f786f9",
        input={ "image": open("path/to/mystery.jpg") }
    )

"an astronaut riding a horse"
```

`vaikerai.run` raises `ModelError` if the prediction fails.
You can access the exception's `prediction` property 
to get more information about the failure.

```python
import vaikerai
from vaikerai.exceptions import ModelError

try:
  output = vaikerai.run("stability-ai/stable-diffusion-3", { "prompt": "An astronaut riding a rainbow unicorn" })
except ModelError as e
  if "(some known issue)" in e.prediction.logs:
    pass

  print("Failed prediction: " + e.prediction.id)
```


## Run a model and stream its output

VaikerAI’s API supports server-sent event streams (SSEs) for language models. 
Use the `stream` method to consume tokens as they're produced by the model.

```python
import vaikerai

for event in vaikerai.stream(
    "meta/meta-llama-3-70b-instruct",
    input={
        "prompt": "Please write a haiku about llamas.",
    },
):
    print(str(event), end="")
```

> [!TIP]
> Some models, like [meta/meta-llama-3-70b-instruct](https://vaikerai.com/meta/meta-llama-3-70b-instruct), 
> don't require a version string. 
> You can always refer to the API documentation on the model page for specifics.

You can also stream the output of a prediction you create.
This is helpful when you want the ID of the prediction separate from its output.

```python
prediction = vaikerai.predictions.create(
    model="meta/meta-llama-3-70b-instruct"
    input={"prompt": "Please write a haiku about llamas."},
    stream=True,
)

for event in prediction.stream():
    print(str(event), end="")
```

For more information, see
["Streaming output"](https://docs.vaikerai.com/streaming) in VaikerAI's docs.


## Run a model in the background

You can start a model and run it in the background:

```python
>>> model = vaikerai.models.get("kvfrans/clipdraw")
>>> version = model.versions.get("5797a99edc939ea0e9242d5e8c9cb3bc7d125b1eac21bda852e5cb79ede2cd9b")
>>> prediction = vaikerai.predictions.create(
    version=version,
    input={"prompt":"Watercolor painting of an underwater submarine"})

>>> prediction
Prediction(...)

>>> prediction.status
'starting'

>>> dict(prediction)
{"id": "...", "status": "starting", ...}

>>> prediction.reload()
>>> prediction.status
'processing'

>>> print(prediction.logs)
iteration: 0, render:loss: -0.6171875
iteration: 10, render:loss: -0.92236328125
iteration: 20, render:loss: -1.197265625
iteration: 30, render:loss: -1.3994140625

>>> prediction.wait()

>>> prediction.status
'succeeded'

>>> prediction.output
'https://.../output.png'
```

## Run a model in the background and get a webhook

You can run a model and get a webhook when it completes, instead of waiting for it to finish:

```python
model = vaikerai.models.get("ai-forever/kandinsky-2.2")
version = model.versions.get("ea1addaab376f4dc227f5368bbd8eff901820fd1cc14ed8cad63b29249e9d463")
prediction = vaikerai.predictions.create(
    version=version,
    input={"prompt":"Watercolor painting of an underwater submarine"},
    webhook="https://example.com/your-webhook",
    webhook_events_filter=["completed"]
)
```

For details on receiving webhooks, see [docs.vaikerai.com/webhooks](https://docs.vaikerai.com/webhooks).

## Compose models into a pipeline

You can run a model and feed the output into another model:

```python
laionide = vaikerai.models.get("afiaka87/laionide-v4").versions.get("b21cbe271e65c1718f2999b038c18b45e21e4fba961181fbfae9342fc53b9e05")
swinir = vaikerai.models.get("jingyunliang/swinir").versions.get("660d922d33153019e8c263a3bba265de882e7f4f70396546b6c9c8f9d47a021a")
image = laionide.predict(prompt="avocado armchair")
upscaled_image = swinir.predict(image=image)
```

## Get output from a running model

Run a model and get its output while it's running:

```python
iterator = vaikerai.run(
    "pixray/text2image:5c347a4bfa1d4523a58ae614c2194e15f2ae682b57e3797a5bb468920aa70ebf",
    input={"prompts": "san francisco sunset"}
)

for image in iterator:
    display(image)
```

## Cancel a prediction

You can cancel a running prediction:

```python
>>> model = vaikerai.models.get("kvfrans/clipdraw")
>>> version = model.versions.get("5797a99edc939ea0e9242d5e8c9cb3bc7d125b1eac21bda852e5cb79ede2cd9b")
>>> prediction = vaikerai.predictions.create(
        version=version,
        input={"prompt":"Watercolor painting of an underwater submarine"}
    )

>>> prediction.status
'starting'

>>> prediction.cancel()

>>> prediction.reload()
>>> prediction.status
'canceled'
```

## List predictions

You can list all the predictions you've run:

```python
vaikerai.predictions.list()
# [<Prediction: 8b0ba5ab4d85>, <Prediction: 494900564e8c>]
```

Lists of predictions are paginated. You can get the next page of predictions by passing the `next` property as an argument to the `list` method:

```python
page1 = vaikerai.predictions.list()

if page1.next:
    page2 = vaikerai.predictions.list(page1.next)
```

## Load output files

Output files are returned as HTTPS URLs. You can load an output file as a buffer:

```python
import vaikerai
from PIL import Image
from urllib.request import urlretrieve

out = vaikerai.run(
    "stability-ai/stable-diffusion:27b93a2413e7f36cd83da926f3656280b2931564ff050bf9575f1fdf9bcd7478",
    input={"prompt": "wavy colorful abstract patterns, oceans"}
    )

urlretrieve(out[0], "/tmp/out.png")
background = Image.open("/tmp/out.png")
```

## List models

You can list the models you've created:

```python
vaikerai.models.list()
```

Lists of models are paginated. You can get the next page of models by passing the `next` property as an argument to the `list` method, or you can use the `paginate` method to fetch pages automatically.

```python
# Automatic pagination using `vaikerai.paginate` (recommended)
models = []
for page in vaikerai.paginate(vaikerai.models.list):
    models.extend(page.results)
    if len(models) > 100:
        break

# Manual pagination using `next` cursors
page = vaikerai.models.list()
while page:
    models.extend(page.results)
    if len(models) > 100:
          break
    page = vaikerai.models.list(page.next) if page.next else None
```

You can also find collections of featured models on VaikerAI:

```python
>>> collections = [collection for page in vaikerai.paginate(vaikerai.collections.list) for collection in page]
>>> collections[0].slug
"vision-models"
>>> collections[0].description
"Multimodal large language models with vision capabilities like object detection and optical character recognition (OCR)"

>>> vaikerai.collections.get("text-to-image").models
[<Model: stability-ai/sdxl>, ...]
```

## Create a model

You can create a model for a user or organization
with a given name, visibility, and hardware SKU:

```python
import vaikerai

model = vaikerai.models.create(
    owner="your-username",
    name="my-model",
    visibility="public",
    hardware="gpu-a40-large"
)
```

Here's how to list of all the available hardware for running models on VaikerAI:

```python
>>> [hw.sku for hw in vaikerai.hardware.list()]
['cpu', 'gpu-t4', 'gpu-a40-small', 'gpu-a40-large']
```

## Fine-tune a model

Use the [training API](https://docs.vaikerai.com/fine-tuning) 
to fine-tune models to make them better at a particular task. 
To see what **language models** currently support fine-tuning, 
check out VaikerAI's [collection of trainable language models](https://vaikerai.com/collections/trainable-language-models).

If you're looking to fine-tune **image models**, 
check out VaikerAI's [guide to fine-tuning image models](https://docs.vaikerai.com/guides/fine-tune-an-image-model).

Here's how to fine-tune a model on VaikerAI:

```python
training = vaikerai.trainings.create(
    model="stability-ai/sdxl",
    version="39ed52f2a78e934b3ba6e2a89f5b1c712de7dfea535525255b1aa35c5565e08b",
    input={
      "input_images": "https://example.com/training-images.zip",
      "token_string": "TOK",
      "caption_prefix": "a photo of TOK",
      "max_train_steps": 1000,
      "use_face_detection_instead": False
    },
    # You need to create a model on VaikerAI that will be the destination for the trained version.
    destination="your-username/model-name"
)
```

## Customize client behavior

The `vaikerai` package exports a default shared client.
This client is initialized with an API token
set by the `VAIKERAI_API_TOKEN` environment variable.

You can create your own client instance to
pass a different API token value,
add custom headers to requests,
or control the behavior of the underlying [HTTPX client](https://www.python-httpx.org/api/#client):

```python
import os
from vaikerai.client import Client

vaikerai = Client(
  api_token=os.environ["SOME_OTHER_VAIKERAI_API_TOKEN"]
  headers={
    "User-Agent": "my-app/1.0
  }
)
```

> [!WARNING]
> Never hardcode authentication credentials like API tokens into your code.
> Instead, pass them as environment variables when running your program.

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md)
