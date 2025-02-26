import os
from typing import Dict, Optional, List, AsyncGenerator, Union, Any
import logging
import json
import random
import time

from fastapi import FastAPI, BackgroundTasks
from starlette.requests import Request
from starlette.responses import StreamingResponse, JSONResponse, Response

from ray import serve
from ray.serve.handle import DeploymentHandle

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.sampling_params import SamplingParams
from vllm.utils import random_uuid

logger = logging.getLogger("ray.serve")

app = FastAPI()

# Function to convert chat messages to prompt text


def messages_to_prompt(messages: List[Dict[str, str]], model_id: str) -> str:
    """Select the appropriate prompt format based on model ID"""
    if "deepseek" in model_id.lower():
        return deepseek_messages_to_prompt(messages)
    elif "mistral" in model_id.lower():
        return mistral_messages_to_prompt(messages)
    else:
        # Default format, can be extended as needed
        return default_messages_to_prompt(messages)


def deepseek_messages_to_prompt(messages: List[Dict[str, str]]) -> str:
    """Format chat messages for DeepSeek models"""
    prompt = ""
    for message in messages:
        role = message["role"]
        content = message["content"]
        if role == "system":
            prompt += f"{content}\n"
        elif role == "user":
            prompt += f"Human: {content}\n"
        elif role == "assistant":
            prompt += f"Assistant: {content}\n"
        else:
            prompt += f"{role.capitalize()}: {content}\n"
    prompt += "Assistant: "
    return prompt


def mistral_messages_to_prompt(messages: List[Dict[str, str]]) -> str:
    """Format chat messages for Mistral models"""
    prompt = ""
    system_message = ""

    # Extract system message
    for message in messages:
        if message["role"] == "system":
            system_message += message["content"] + "\n"

    # Add system message to prompt beginning
    if system_message:
        prompt = f"<s>[INST] {system_message}\n"
    else:
        prompt = "<s>[INST] "

    # Add user and assistant messages
    for i, message in enumerate(messages):
        if message["role"] == "system":
            continue
        elif message["role"] == "user":
            if i > 0 and i-1 < len(messages) and messages[i-1]["role"] == "assistant":
                prompt += f"[/INST]\n\n[INST] {message['content']}"
            else:
                prompt += message["content"]
        elif message["role"] == "assistant":
            prompt += f" [/INST]\n\n{message['content']}\n\n"

    # Make sure prompt ends with user message for assistant's response
    if messages[-1]["role"] == "user":
        prompt += " [/INST]\n\n"

    return prompt


def default_messages_to_prompt(messages: List[Dict[str, str]]) -> str:
    """Generic chat message formatting"""
    prompt = ""
    for message in messages:
        role = message["role"]
        content = message["content"]
        prompt += f"{role.capitalize()}: {content}\n"
    prompt += "Assistant: "
    return prompt

# Helper to create a chat completion response


def create_chat_completion_response(
    request_id: str,
    model_id: str,
    content: str,
    prompt_tokens: int,
    completion_tokens: int,
) -> dict:
    """Create a standardized chat completion response dictionary"""
    return {
        "id": request_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_id,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content
                },
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens
        }
    }


@serve.deployment(name="VLLMDeployment")
class VLLMDeployment:
    def __init__(self, **kwargs):
        """
        Construct a VLLM deployment.

        For full parameter list, see:
        https://github.com/vllm-project/vllm/blob/main/vllm/engine/arg_utils.py
        """
        self.model_id = kwargs.get("model")
        args = AsyncEngineArgs(**kwargs)
        # Fix for CUDA device visibility issue
        # https://github.com/vllm-project/vllm/issues/8402#issuecomment-2489432973
        if 'CUDA_VISIBLE_DEVICES' in os.environ:
            del os.environ['CUDA_VISIBLE_DEVICES']
        self.engine = AsyncLLMEngine.from_engine_args(args)

    async def stream_chat_response(self, request_output_generator, model_id: str) -> AsyncGenerator[str, None]:
        """Generate streaming chat completion response"""
        # First chunk with assistant role
        chunk = {
            "id": random_uuid(),
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model_id,
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant"},
                    "finish_reason": None
                }
            ]
        }
        yield f"data: {json.dumps(chunk)}\n\n"

        # Stream content chunks
        async for request_output in request_output_generator:
            if len(request_output.outputs) == 0:
                continue

            delta_text = request_output.outputs[0].text
            chunk = {
                "id": random_uuid(),
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model_id,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": delta_text},
                        "finish_reason": None
                    }
                ]
            }
            yield f"data: {json.dumps(chunk)}\n\n"

        # Final chunk signaling completion
        chunk = {
            "id": random_uuid(),
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model_id,
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop"
                }
            ]
        }
        yield f"data: {json.dumps(chunk)}\n\n"
        yield "data: [DONE]\n\n"

    async def abort_request_on_disconnect(self, request_id: str) -> None:
        """Abort request when client disconnects"""
        try:
            await self.engine.abort(request_id)
        except Exception as e:
            logger.error(f"Error aborting request {request_id}: {e}")

    async def handle_chat_request(self, request: dict, model_id: str) -> Union[dict, StreamingResponse]:
        """Process a chat completion request"""
        request_id = random_uuid()

        # Extract messages from request
        messages = request.get("messages", [])
        if not messages:
            raise ValueError("Messages array cannot be empty")

        # Convert messages to prompt text
        prompt = messages_to_prompt(messages, model_id)

        # Extract sampling parameters
        sampling_params = SamplingParams(
            temperature=request.get("temperature", 1.0),
            top_p=request.get("top_p", 1.0),
            max_tokens=request.get("max_tokens", 1024),
            stop=request.get("stop"),
            frequency_penalty=request.get("frequency_penalty", 0.0),
            presence_penalty=request.get("presence_penalty", 0.0),
        )

        # Check if streaming is requested
        stream = request.get("stream", False)

        # Get generation results
        results_generator = self.engine.generate(
            prompt, sampling_params, request_id)

        if stream:
            # Return streaming response
            background_tasks = BackgroundTasks()
            background_tasks.add_task(
                self.abort_request_on_disconnect, request_id)
            return StreamingResponse(
                self.stream_chat_response(results_generator, model_id),
                media_type="text/event-stream",
                background=background_tasks,
            )

        # Non-streaming response
        final_output = None
        async for request_output in results_generator:
            final_output = request_output

        if not final_output or not final_output.outputs:
            return {
                "error": {
                    "message": "Failed to generate text",
                    "type": "server_error",
                    "code": 500
                }
            }

        generated_text = final_output.outputs[0].text

        # Calculate token usage
        prompt_tokens = len(final_output.prompt_token_ids)
        completion_tokens = len(
            final_output.outputs[0].token_ids) if final_output.outputs else 0

        # Create and return complete response
        return create_chat_completion_response(
            request_id=request_id,
            model_id=model_id,
            content=generated_text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

    async def __call__(self, request_dict: dict) -> Any:
        """Handle API requests"""
        try:
            response = await self.handle_chat_request(request_dict, self.model_id)

            # If response is already a StreamingResponse, return it directly
            if isinstance(response, StreamingResponse):
                return response

            # Otherwise return a JSON response
            return JSONResponse(content=response)

        except Exception as e:
            logger.exception(f"Error processing request: {e}")
            error_response = {
                "error": {
                    "message": str(e),
                    "type": "invalid_request_error",
                    "code": 400
                }
            }
            return JSONResponse(content=error_response, status_code=400)


@serve.deployment
@serve.ingress(app)
class MultiModelDeployment:
    def __init__(self, models: Dict[str, DeploymentHandle]):
        self.models = models
        # Create mapping from model IDs to friendly names
        self.model_id_to_name = {
            "Valdemardi/DeepSeek-R1-Distill-Llama-70B-AWQ": "deepseek-r1-70b",
            "stelterlab/Mistral-Small-24B-Instruct-2501-AWQ": "mistral-small-24b",
        }
        # Reverse mapping for lookups
        self.name_to_model_id = {v: k for k,
                                 v in self.model_id_to_name.items()}

    @app.get("/v1/models")
    async def list_models(self):
        """List available models, compatible with OpenAI API"""
        available_models = []
        for model_id in self.models.keys():
            model_name = self.model_id_to_name.get(model_id, model_id)
            available_models.append({
                "id": model_name,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "kuberay-user",
            })
        return {"data": available_models, "object": "list"}

    @app.post("/v1/chat/completions")
    async def create_chat_completion(self, request: Request):
        """Create chat completion, compatible with OpenAI API"""
        try:
            model_request = await request.json()

            # Get model ID from request
            requested_model = model_request.get("model", "")

            # Get model ID from header, which takes precedence
            header_model_id = request.headers.get("Model-ID")

            # Decide which model to use
            if header_model_id and header_model_id in self.models:
                model_id = header_model_id
            elif requested_model in self.name_to_model_id:
                # Convert friendly name to full ID
                model_id = self.name_to_model_id[requested_model]
            elif requested_model in self.models:
                # User provided the full ID directly
                model_id = requested_model
            elif not header_model_id and not requested_model:
                # No model specified, choose randomly
                model_id = random.choice(list(self.models.keys()))
            else:
                # Invalid model ID
                error_message = {
                    "error": {
                        "message": f"Model '{requested_model or header_model_id}' not found",
                        "type": "invalid_request_error",
                        "code": "model_not_found",
                    }
                }
                return JSONResponse(status_code=404, content=error_message)

            logger.info(f"Using model ID: {model_id}")
            model_handle = self.models[model_id]

            # Pass request to the appropriate model handler
            response = await model_handle.remote(model_request)

            # Pass through the response
            return response

        except Exception as e:
            logger.exception(f"Error handling chat completion request: {e}")
            error_response = {
                "error": {
                    "message": str(e),
                    "type": "internal_server_error",
                    "code": 500,
                }
            }
            return JSONResponse(status_code=500, content=error_response)

    @app.get("/health")
    async def health_check(self):
        """Health check endpoint"""
        return {"status": "healthy"}


def build_app() -> serve.Application:
    """Build a Serve application with multiple models."""

    models_handles = {}

    # Model 1: DeepSeek
    model_1_id = "Valdemardi/DeepSeek-R1-Distill-Llama-70B-AWQ"
    model_1_kwargs = {
        "model": model_1_id,
        "tensor_parallel_size": 4,
        "quantization": "awq",
        "dtype": "half",  # Use FP16 for faster inference
        "gpu_memory_utilization": 0.85,  # Control GPU memory usage
        "max_model_len": 80960,  # Maximum token length
        "max_num_seqs": 32,  # Maximum sequences per iteration
        "trust_remote_code": True,  # Trust remote code if needed by model
    }
    models_handles[model_1_id] = VLLMDeployment.options(
        ray_actor_options={"num_cpus": 4, "num_gpus": 4}).bind(**model_1_kwargs)

    # Model 2: Mistral
    model_2_id = "stelterlab/Mistral-Small-24B-Instruct-2501-AWQ"
    model_2_kwargs = {
        "model": model_2_id,
        "tensor_parallel_size": 2,
        "quantization": "awq",
        "dtype": "half",
        "gpu_memory_utilization": 0.85,
        "max_num_seqs": 32,
        "trust_remote_code": True,
    }
    models_handles[model_2_id] = VLLMDeployment.options(
        ray_actor_options={"num_cpus": 4, "num_gpus": 2}).bind(**model_2_kwargs)

    # Create and return multi-model deployment
    return MultiModelDeployment.bind(models_handles)


# Application exposed in Ray Serve
multi_model = build_app()
