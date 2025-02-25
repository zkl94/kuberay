import os
from typing import Dict, List, AsyncGenerator, Union, Any
import logging
import json
import time
import asyncio

from fastapi import FastAPI, BackgroundTasks, HTTPException
from starlette.requests import Request
from starlette.responses import StreamingResponse, JSONResponse

from ray import serve

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.sampling_params import SamplingParams
from vllm.utils import random_uuid

# Set up proper logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ray.serve")

# ==========================================================
# Common Utility Functions
# ==========================================================


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

# ==========================================================
# Prompt Formatting Functions
# ==========================================================


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

# ==========================================================
# DeepSeek LLM Deployment
# ==========================================================


deepseek_app = FastAPI()


@serve.deployment(name="DeepSeekDeployment")
@serve.ingress(deepseek_app)
class DeepSeekDeployment:
    def __init__(self):
        """Initialize the DeepSeek model"""
        # Model configuration
        self.model_id = "Valdemardi/DeepSeek-R1-Distill-Qwen-32B-AWQ"
        self.friendly_name = "deepseek-qwen-32b"

        # Engine configuration
        engine_args = {
            "model": self.model_id,
            "tensor_parallel_size": 4,
            "quantization": "awq",
            "dtype": "half",
            "gpu_memory_utilization": 0.90,
            "max_model_len": 16384,
            "max_num_seqs": 16,
            "trust_remote_code": True,
        }

        # Fix for CUDA device visibility issue
        if 'CUDA_VISIBLE_DEVICES' in os.environ:
            del os.environ['CUDA_VISIBLE_DEVICES']

        # Initialize the engine
        logger.info(f"Initializing DeepSeek model: {self.model_id}")
        args = AsyncEngineArgs(**engine_args)
        self.engine = AsyncLLMEngine.from_engine_args(args)
        logger.info(f"DeepSeek model {self.model_id} initialized successfully")

        # Set reasonable timeout
        self.request_timeout = 300  # 5 minutes

    async def abort_request_on_disconnect(self, request_id: str) -> None:
        """Abort request when client disconnects"""
        try:
            logger.info(
                f"Aborting request {request_id} due to client disconnect")
            await self.engine.abort(request_id)
        except Exception as e:
            logger.error(f"Error aborting request {request_id}: {e}")

    async def stream_chat_response(self, request_output_generator, model_id: str) -> AsyncGenerator[str, None]:
        """Generate streaming chat completion response"""
        try:
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
            previous_text = ""
            async for request_output in request_output_generator:
                if len(request_output.outputs) == 0:
                    continue

                # Get the new delta text (only what's new since last chunk)
                current_text = request_output.outputs[0].text
                delta_text = current_text[len(previous_text):]
                previous_text = current_text

                # Only send non-empty chunks
                if delta_text:
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

                    # Add a small sleep to avoid overwhelming the client
                    await asyncio.sleep(0.01)

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

        except Exception as e:
            logger.exception(f"Error in streaming: {str(e)}")
            error_chunk = {
                "error": {
                    "message": f"Streaming error: {str(e)}",
                    "type": "server_error",
                    "code": 500
                }
            }
            yield f"data: {json.dumps(error_chunk)}\n\n"
            yield "data: [DONE]\n\n"

    @deepseek_app.get("/models")
    async def list_models(self):
        """List available models, compatible with OpenAI API"""
        return {
            "data": [
                {
                    "id": self.friendly_name,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "kuberay-user",
                }
            ],
            "object": "list"
        }

    @deepseek_app.post("/chat/completions")
    async def create_chat_completion(self, request: Request):
        """Create chat completion API, compatible with OpenAI API"""
        try:
            request_dict = await request.json()
            request_id = random_uuid()

            # Extract messages from request
            messages = request_dict.get("messages", [])
            if not messages:
                raise HTTPException(
                    status_code=400, detail="Messages array cannot be empty")

            # Convert messages to prompt text
            prompt = deepseek_messages_to_prompt(messages)

            # Extract sampling parameters
            sampling_params = SamplingParams(
                temperature=request_dict.get("temperature", 0.7),
                top_p=request_dict.get("top_p", 0.9),
                max_tokens=min(request_dict.get("max_tokens", 1024), 4096),
                stop=request_dict.get("stop"),
                frequency_penalty=request_dict.get("frequency_penalty", 0.0),
                presence_penalty=request_dict.get("presence_penalty", 0.0),
            )

            # Check if streaming is requested
            stream = request_dict.get("stream", False)

            # Get generation results
            results_generator = self.engine.generate(
                prompt, sampling_params, request_id)

            if stream:
                # Return streaming response
                background_tasks = BackgroundTasks()
                background_tasks.add_task(
                    self.abort_request_on_disconnect, request_id)

                return StreamingResponse(
                    self.stream_chat_response(
                        results_generator, self.friendly_name),
                    media_type="text/event-stream",
                    background=background_tasks,
                )

            # Non-streaming response
            final_output = None
            async for request_output in results_generator:
                final_output = request_output

            if not final_output or not final_output.outputs:
                return JSONResponse(
                    status_code=500,
                    content={
                        "error": {
                            "message": "Failed to generate text",
                            "type": "server_error",
                            "code": 500
                        }
                    }
                )

            generated_text = final_output.outputs[0].text

            # Calculate token usage
            prompt_tokens = len(final_output.prompt_token_ids)
            completion_tokens = len(
                final_output.outputs[0].token_ids) if final_output.outputs else 0

            # Create and return complete response
            response = create_chat_completion_response(
                request_id=request_id,
                model_id=self.friendly_name,
                content=generated_text,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )

            return JSONResponse(content=response)

        except Exception as e:
            logger.exception(f"Error processing request: {e}")
            return JSONResponse(
                status_code=500,
                content={
                    "error": {
                        "message": str(e),
                        "type": "server_error",
                        "code": 500
                    }
                }
            )

    @deepseek_app.get("/health")
    async def health_check(self):
        """Health check endpoint"""
        return {"status": "healthy", "model": self.friendly_name}

# ==========================================================
# Mistral LLM Deployment
# ==========================================================


mistral_app = FastAPI()


@serve.deployment(name="MistralDeployment")
@serve.ingress(mistral_app)
class MistralDeployment:
    def __init__(self):
        """Initialize the Mistral model"""
        # Model configuration
        self.model_id = "casperhansen/mistral-small-24b-instruct-2501-awq"
        self.friendly_name = "mistral-small-24b"

        # Engine configuration
        engine_args = {
            "model": self.model_id,
            "tensor_parallel_size": 2,
            "quantization": "awq",
            "dtype": "half",
            "gpu_memory_utilization": 0.90,
            "max_model_len": 16384,
            "max_num_seqs": 16,
            "trust_remote_code": True,
        }

        # Fix for CUDA device visibility issue
        if 'CUDA_VISIBLE_DEVICES' in os.environ:
            del os.environ['CUDA_VISIBLE_DEVICES']

        # Initialize the engine
        logger.info(f"Initializing Mistral model: {self.model_id}")
        args = AsyncEngineArgs(**engine_args)
        self.engine = AsyncLLMEngine.from_engine_args(args)
        logger.info(f"Mistral model {self.model_id} initialized successfully")

        # Set reasonable timeout
        self.request_timeout = 300  # 5 minutes

    async def abort_request_on_disconnect(self, request_id: str) -> None:
        """Abort request when client disconnects"""
        try:
            logger.info(
                f"Aborting request {request_id} due to client disconnect")
            await self.engine.abort(request_id)
        except Exception as e:
            logger.error(f"Error aborting request {request_id}: {e}")

    async def stream_chat_response(self, request_output_generator, model_id: str) -> AsyncGenerator[str, None]:
        """Generate streaming chat completion response"""
        try:
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
            previous_text = ""
            async for request_output in request_output_generator:
                if len(request_output.outputs) == 0:
                    continue

                # Get the new delta text (only what's new since last chunk)
                current_text = request_output.outputs[0].text
                delta_text = current_text[len(previous_text):]
                previous_text = current_text

                # Only send non-empty chunks
                if delta_text:
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

                    # Add a small sleep to avoid overwhelming the client
                    await asyncio.sleep(0.01)

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

        except Exception as e:
            logger.exception(f"Error in streaming: {str(e)}")
            error_chunk = {
                "error": {
                    "message": f"Streaming error: {str(e)}",
                    "type": "server_error",
                    "code": 500
                }
            }
            yield f"data: {json.dumps(error_chunk)}\n\n"
            yield "data: [DONE]\n\n"

    @mistral_app.get("/models")
    async def list_models(self):
        """List available models, compatible with OpenAI API"""
        return {
            "data": [
                {
                    "id": self.friendly_name,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "kuberay-user",
                }
            ],
            "object": "list"
        }

    @mistral_app.post("/chat/completions")
    async def create_chat_completion(self, request: Request):
        """Create chat completion API, compatible with OpenAI API"""
        try:
            request_dict = await request.json()
            request_id = random_uuid()

            # Extract messages from request
            messages = request_dict.get("messages", [])
            if not messages:
                raise HTTPException(
                    status_code=400, detail="Messages array cannot be empty")

            # Convert messages to prompt text
            prompt = mistral_messages_to_prompt(messages)

            # Extract sampling parameters
            sampling_params = SamplingParams(
                temperature=request_dict.get("temperature", 0.7),
                top_p=request_dict.get("top_p", 0.9),
                max_tokens=min(request_dict.get("max_tokens", 1024), 4096),
                stop=request_dict.get("stop"),
                frequency_penalty=request_dict.get("frequency_penalty", 0.0),
                presence_penalty=request_dict.get("presence_penalty", 0.0),
            )

            # Check if streaming is requested
            stream = request_dict.get("stream", False)

            # Get generation results
            results_generator = self.engine.generate(
                prompt, sampling_params, request_id)

            if stream:
                # Return streaming response
                background_tasks = BackgroundTasks()
                background_tasks.add_task(
                    self.abort_request_on_disconnect, request_id)

                return StreamingResponse(
                    self.stream_chat_response(
                        results_generator, self.friendly_name),
                    media_type="text/event-stream",
                    background=background_tasks,
                )

            # Non-streaming response
            final_output = None
            async for request_output in results_generator:
                final_output = request_output

            if not final_output or not final_output.outputs:
                return JSONResponse(
                    status_code=500,
                    content={
                        "error": {
                            "message": "Failed to generate text",
                            "type": "server_error",
                            "code": 500
                        }
                    }
                )

            generated_text = final_output.outputs[0].text

            # Calculate token usage
            prompt_tokens = len(final_output.prompt_token_ids)
            completion_tokens = len(
                final_output.outputs[0].token_ids) if final_output.outputs else 0

            # Create and return complete response
            response = create_chat_completion_response(
                request_id=request_id,
                model_id=self.friendly_name,
                content=generated_text,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )

            return JSONResponse(content=response)

        except Exception as e:
            logger.exception(f"Error processing request: {e}")
            return JSONResponse(
                status_code=500,
                content={
                    "error": {
                        "message": str(e),
                        "type": "server_error",
                        "code": 500
                    }
                }
            )

    @mistral_app.get("/health")
    async def health_check(self):
        """Health check endpoint"""
        return {"status": "healthy", "model": self.friendly_name}

# ==========================================================
# Root API App for Model Discovery (Optional)
# ==========================================================


root_app = FastAPI()


@serve.deployment(name="ModelDiscovery")
@serve.ingress(root_app)
class ModelDiscovery:
    @root_app.get("/v1/models")
    async def list_all_models(self):
        """List all available models across deployments"""
        return {
            "data": [
                {
                    "id": "deepseek-qwen-32b",
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "kuberay-user",
                },
                {
                    "id": "mistral-small-24b",
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "kuberay-user",
                }
            ],
            "object": "list"
        }

    @root_app.get("/health")
    async def health_check(self):
        """Root health check endpoint"""
        return {"status": "healthy", "service": "llm-cluster"}

# ==========================================================
# Build Application Function for Ray Serve
# ==========================================================


def build_app(args: dict = None):
    """Build and return deployments for Ray Serve

    Args:
        args: A dictionary of arguments from Ray Serve (required parameter)

    Returns:
        A list of Ray Serve deployments
    """
    # Create model deployments
    deepseek_deployment = DeepSeekDeployment.bind()
    mistral_deployment = MistralDeployment.bind()

    # The model discovery deployment is optional but useful for service discovery
    discovery_deployment = ModelDiscovery.bind()

    # Return all deployments in a list
    return [deepseek_deployment, mistral_deployment, discovery_deployment]
