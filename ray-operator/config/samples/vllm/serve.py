import os

from typing import Dict, Optional, List
import logging
import json
import os
import random
from typing import AsyncGenerator


from fastapi import FastAPI
from starlette.requests import Request
from starlette.responses import StreamingResponse, JSONResponse, Response

from ray import serve
from ray.serve.handle import DeploymentHandle

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.entrypoints.openai.cli_args import make_arg_parser
from vllm.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ErrorResponse,
)
from vllm.sampling_params import SamplingParams
from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
from vllm.entrypoints.openai.serving_engine import LoRAModulePath, PromptAdapterPath
from vllm.utils import FlexibleArgumentParser
from vllm.entrypoints.logger import RequestLogger
from vllm.utils import random_uuid
from fastapi import BackgroundTasks

logger = logging.getLogger("ray.serve")

app = FastAPI()


@serve.deployment(name="VLLMDeployment")
class VLLMDeployment:
    def __init__(self, **kwargs):
        """
        Construct a VLLM deployment.

        Refer to https://github.com/vllm-project/vllm/blob/main/vllm/engine/arg_utils.py
        for the full list of arguments.

        Args:
            model: name or path of the huggingface model to use
            download_dir: directory to download and load the weights,
                default to the default cache dir of huggingface.
            use_np_weights: save a numpy copy of model weights for
                faster loading. This can increase the disk usage by up to 2x.
            use_dummy_weights: use dummy values for model weights.
            dtype: data type for model weights and activations.
                The "auto" option will use FP16 precision
                for FP32 and FP16 models, and BF16 precision.
                for BF16 models.
            seed: random seed.
            worker_use_ray: use Ray for distributed serving, will be
                automatically set when using more than 1 GPU
            pipeline_parallel_size: number of pipeline stages.
            tensor_parallel_size: number of tensor parallel replicas.
            block_size: token block size.
            swap_space: CPU swap space size (GiB) per GPU.
            gpu_memory_utilization: the percentage of GPU memory to be used for
                the model executor
            max_num_batched_tokens: maximum number of batched tokens per iteration
            max_num_seqs: maximum number of sequences per iteration.
            disable_log_stats: disable logging statistics.
            engine_use_ray: use Ray to start the LLM engine in a separate
                process as the server process.
            disable_log_requests: disable logging requests.
        """
        args = AsyncEngineArgs(**kwargs)
        # https://github.com/vllm-project/vllm/issues/8402#issuecomment-2489432973
        del os.environ['CUDA_VISIBLE_DEVICES']
        self.engine = AsyncLLMEngine.from_engine_args(args)

    async def stream_results(self, results_generator) -> AsyncGenerator[bytes, None]:
        num_returned = 0
        async for request_output in results_generator:
            text_outputs = [output.text for output in request_output.outputs]
            assert len(text_outputs) == 1
            text_output = text_outputs[0][num_returned:]
            ret = {"text": text_output}
            yield (json.dumps(ret) + "\n").encode("utf-8")
            num_returned += len(text_output)

    async def may_abort_request(self, request_id) -> None:
        await self.engine.abort(request_id)

    async def __call__(self, request_dict: dict) -> str:
        """Generate completion for the request.

        The request should be a JSON object with the following fields:
        - prompt: the prompt to use for the generation.
        - stream: whether to stream the results or not.
        - other fields: the sampling parameters (See `SamplingParams` for details).
        """
        # request_dict = await request.json()
        prompt = request_dict.pop("prompt")
        stream = request_dict.pop("stream", False)
        sampling_params = SamplingParams(**request_dict)
        request_id = random_uuid()
        results_generator = self.engine.generate(
            prompt, sampling_params, request_id)
        if stream:
            background_tasks = BackgroundTasks()
            # Using background_taks to abort the the request
            # if the client disconnects.
            background_tasks.add_task(self.may_abort_request, request_id)
            return StreamingResponse(
                self.stream_results(results_generator), background=background_tasks
            )

        final_output = None
        async for request_output in results_generator:
            final_output = request_output

        assert final_output is not None
        prompt = final_output.prompt
        text_outputs = [
            prompt + output.text for output in final_output.outputs]
        ret = {"text": text_outputs}
        return json.dumps(ret)


@serve.deployment
@serve.ingress(app)
class MultiModelDeployment:
    def __init__(self, models: Dict[str, DeploymentHandle]):
        self.models = models

    @app.post("/v1/chat/completions")
    async def create_chat_completion(self, request: Request):
        model_request = await request.json()
        # or serve.get_multiplexed_model_id() if you set multiplexed_model_id in ray serve config
        model_id = request.headers.get("Model-ID")
        if model_id in self.models:
            # pass request to create_chat_completion
            response = await self.models[model_id].create_chat_completion.remote(model_request, request)
        elif not model_id:
            model_id = random.choice(list(self.models.keys()))
            # pass request to create_chat_completion
            response = await self.models[model_id].create_chat_completion.remote(model_request, request)
        else:
            return Response(status_code=400, content=json.dumps({"message": "invalid model ID"}))

        return Response(content=response)


def build_app() -> serve.Application:
    """Builds the Serve app with multiple models."""

    models_handles = {}

    # Model 1
    model_1_kwargs = {
        "model": os.environ['MODEL_1_ID'],
        "tensor_parallel_size": int(os.environ['MODEL_1_TENSOR_PARALLELISM']),
        # optional quantization
        # "quantization": os.environ.get('MODEL_1_QUANTIZE', None),
        "quantization": "awq",
    }
    models_handles[os.environ['MODEL_1_ID']] = VLLMDeployment.options(
        ray_actor_options={"num_cpus": 4}).bind(**model_1_kwargs)

    # Model 2
    model_2_kwargs = {
        "model": os.environ['MODEL_2_ID'],
        "tensor_parallel_size": int(os.environ['MODEL_2_TENSOR_PARALLELISM']),
        # optional quantization
        # "quantization":  os.environ.get('MODEL_2_QUANTIZE', None),
        "quantization": "awq",
    }
    models_handles[os.environ['MODEL_2_ID']] = VLLMDeployment.options(
        ray_actor_options={"num_cpus": 4}).bind(**model_2_kwargs)

    return MultiModelDeployment.bind(models_handles)


multi_model = build_app()
