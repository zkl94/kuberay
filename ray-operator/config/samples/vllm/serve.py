import os
from typing import Dict, Optional, List, AsyncGenerator, Union, Any
import logging
import json
import random
import time
from dataclasses import asdict

from fastapi import FastAPI, HTTPException, BackgroundTasks
from starlette.requests import Request
from starlette.responses import StreamingResponse, JSONResponse, Response

from ray import serve
from ray.serve.handle import DeploymentHandle

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.sampling_params import SamplingParams
from vllm.utils import random_uuid

# OpenAI API类型定义，用于兼容OpenAI API
from vllm.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatCompletionResponseStreamChoice,
    ChatCompletionStreamResponse,
    ChatMessage,
    DeltaMessage,
    FunctionCall,
    ModelCard,
    ModelList,
    ErrorResponse,
    UsageInfo,
)

logger = logging.getLogger("ray.serve")

app = FastAPI()

# 转换聊天消息为提示文本的函数
def messages_to_prompt(messages: List[Dict[str, str]], model_id: str) -> str:
    """根据模型ID选择适当的提示格式转换函数"""
    if "deepseek" in model_id.lower():
        return deepseek_messages_to_prompt(messages)
    elif "mistral" in model_id.lower():
        return mistral_messages_to_prompt(messages)
    else:
        # 默认格式，可根据需要扩展
        return default_messages_to_prompt(messages)

def deepseek_messages_to_prompt(messages: List[Dict[str, str]]) -> str:
    """为DeepSeek模型格式化聊天消息"""
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
    """为Mistral模型格式化聊天消息"""
    prompt = ""
    system_message = ""
    
    # 提取系统消息
    for message in messages:
        if message["role"] == "system":
            system_message += message["content"] + "\n"
    
    # 添加系统消息到提示前部
    if system_message:
        prompt = f"<s>[INST] {system_message}\n"
    else:
        prompt = "<s>[INST] "
    
    # 添加用户和助手消息
    for i, message in enumerate(messages):
        if message["role"] == "system":
            continue
        elif message["role"] == "user":
            if i > 0 and messages[i-1]["role"] == "assistant":
                prompt += f"[/INST]\n\n[INST] {message['content']}"
            else:
                prompt += message["content"]
        elif message["role"] == "assistant":
            prompt += f" [/INST]\n\n{message['content']}\n\n"
    
    # 确保提示以用户消息结尾，并为助手添加响应前缀
    if messages[-1]["role"] == "user":
        prompt += " [/INST]\n\n"
    
    return prompt

def default_messages_to_prompt(messages: List[Dict[str, str]]) -> str:
    """通用聊天消息格式化"""
    prompt = ""
    for message in messages:
        role = message["role"]
        content = message["content"]
        prompt += f"{role.capitalize()}: {content}\n"
    prompt += "Assistant: "
    return prompt

@serve.deployment(name="VLLMDeployment")
class VLLMDeployment:
    def __init__(self, **kwargs):
        """
        构建一个VLLM部署。
        
        详细参数请参考：https://github.com/vllm-project/vllm/blob/main/vllm/engine/arg_utils.py
        """
        self.model_id = kwargs.get("model")
        args = AsyncEngineArgs(**kwargs)
        # https://github.com/vllm-project/vllm/issues/8402#issuecomment-2489432973
        if 'CUDA_VISIBLE_DEVICES' in os.environ:
            del os.environ['CUDA_VISIBLE_DEVICES']
        self.engine = AsyncLLMEngine.from_engine_args(args)

    async def stream_completion(self, request_output_generator, model_id: str) -> AsyncGenerator[str, None]:
        """生成流式完成响应"""
        choice_data = ChatCompletionResponseStreamChoice(
            index=0, delta=DeltaMessage(role="assistant"), finish_reason=None
        )
        chunk = ChatCompletionStreamResponse(
            id=random_uuid(),
            model=model_id,
            choices=[choice_data],
            created=int(time.time()),
            object="chat.completion.chunk",
        )
        yield f"data: {json.dumps(asdict(chunk))}\n\n"

        async for request_output in request_output_generator:
            if len(request_output.outputs) == 0:
                continue
                
            delta_text = request_output.outputs[0].text
            choice_data = ChatCompletionResponseStreamChoice(
                index=0, delta=DeltaMessage(content=delta_text), finish_reason=None
            )
            chunk = ChatCompletionStreamResponse(
                id=random_uuid(),
                model=model_id,
                choices=[choice_data],
                created=int(time.time()),
                object="chat.completion.chunk",
            )
            yield f"data: {json.dumps(asdict(chunk))}\n\n"

        # 发送最终完成信号
        choice_data = ChatCompletionResponseStreamChoice(
            index=0, delta=DeltaMessage(), finish_reason="stop"
        )
        chunk = ChatCompletionStreamResponse(
            id=random_uuid(),
            model=model_id,
            choices=[choice_data],
            created=int(time.time()),
            object="chat.completion.chunk",
        )
        yield f"data: {json.dumps(asdict(chunk))}\n\n"
        yield "data: [DONE]\n\n"

    async def handle_chat_request(self, request: ChatCompletionRequest, model_id: str) -> Union[ChatCompletionResponse, StreamingResponse]:
        """处理聊天完成请求"""
        request_id = random_uuid()
        
        # 从请求中提取消息和参数
        # 处理不同类型的消息对象
        messages = []
        for msg in request.messages:
            if hasattr(msg, 'dict') and callable(msg.dict):
                # 如果是Pydantic模型或有dict()方法的对象
                messages.append(msg.dict())
            elif isinstance(msg, dict):
                # 如果已经是字典
                messages.append(msg)
            else:
                # 尝试转换为字典
                try:
                    messages.append({"role": msg.role, "content": msg.content})
                except AttributeError:
                    logger.error(f"无法处理消息格式: {msg}")
                    raise ValueError(f"消息格式无效: {msg}")
        
        prompt = messages_to_prompt(messages, model_id)
        
        # 提取采样参数
        sampling_params = SamplingParams(
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=request.max_tokens or 1024,
            stop=request.stop,
            frequency_penalty=getattr(request, "frequency_penalty", 0.0),
            presence_penalty=getattr(request, "presence_penalty", 0.0),
        )
        
        # 获取生成结果
        results_generator = self.engine.generate(prompt, sampling_params, request_id)
        
        if request.stream:
            # 返回流式响应
            background_tasks = BackgroundTasks()
            background_tasks.add_task(self.abort_request_on_disconnect, request_id)
            return StreamingResponse(
                self.stream_completion(results_generator, model_id),
                media_type="text/event-stream",
                background=background_tasks,
            )
        
        # 非流式响应
        final_output = None
        async for request_output in results_generator:
            if not final_output:
                final_output = request_output
            else:
                final_output = request_output
                
        generated_text = final_output.outputs[0].text if final_output.outputs else ""
        
        # 计算令牌用量
        prompt_tokens = final_output.prompt_token_ids.shape[0]
        completion_tokens = len(final_output.outputs[0].token_ids) if final_output.outputs else 0
        
        # 构建响应
        choice = ChatCompletionResponseChoice(
            index=0,
            message=ChatMessage(role="assistant", content=generated_text),
            finish_reason="stop",
        )
        
        return ChatCompletionResponse(
            id=request_id,
            model=model_id,
            choices=[choice],
            created=int(time.time()),
            object="chat.completion",
            usage=UsageInfo(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
        )
    
    async def abort_request_on_disconnect(self, request_id: str) -> None:
        """在客户端断开连接时中止请求"""
        await self.engine.abort(request_id)
    
    async def handle_prompt_request(self, prompt: str, params: dict, model_id: str) -> Union[Response, JSONResponse]:
        """处理传统的单一prompt格式请求"""
        request_id = random_uuid()
        
        # 提取采样参数
        sampling_params = SamplingParams(
            temperature=params.get("temperature", 1.0),
            top_p=params.get("top_p", 1.0),
            max_tokens=params.get("max_tokens", 1024),
            stop=params.get("stop", None),
            frequency_penalty=params.get("frequency_penalty", 0.0),
            presence_penalty=params.get("presence_penalty", 0.0),
        )
        
        stream = params.get("stream", False)
        
        # 获取生成结果
        results_generator = self.engine.generate(prompt, sampling_params, request_id)
        
        if stream:
            # 返回流式响应
            background_tasks = BackgroundTasks()
            background_tasks.add_task(self.abort_request_on_disconnect, request_id)
            return StreamingResponse(
                self.stream_completion(results_generator, model_id),
                media_type="text/


@serve.deployment
@serve.ingress(app)
class MultiModelDeployment:
    def __init__(self, models: Dict[str, DeploymentHandle]):
        self.models = models
        # 创建模型ID到用户友好名称的映射
        self.model_id_to_name = {
            "Valdemardi/DeepSeek-R1-Distill-Qwen-32B-AWQ": "deepseek-r1-32b",
            "stelterlab/Mistral-Small-24B-Instruct-2501-AWQ": "mistral-small-24b",
        }
        # 反向映射，方便查找
        self.name_to_model_id = {v: k for k, v in self.model_id_to_name.items()}

    @app.get("/v1/models")
    async def list_models(self):
        """列出可用的模型，兼容OpenAI API"""
        available_models = []
        for model_id in self.models.keys():
            model_name = self.model_id_to_name.get(model_id, model_id)
            available_models.append(
                ModelCard(
                    id=model_name,
                    object="model",
                    created=int(time.time()),
                    owned_by="kuberay-user",
                )
            )
        return ModelList(data=available_models, object="list")

    @app.post("/v1/chat/completions")
    async def create_chat_completion(self, request: Request):
        """创建聊天完成，兼容OpenAI API"""
        try:
            model_request = await request.json()
            
            # 从请求中获取模型ID
            requested_model = model_request.get("model", "")
            
            # 从请求头获取模型ID，优先级高于请求中的model字段
            header_model_id = request.headers.get("Model-ID")
            
            # 决定使用哪个模型
            if header_model_id and header_model_id in self.models:
                model_id = header_model_id
            elif requested_model in self.name_to_model_id:
                # 如果用户使用简短名称，转换为完整ID
                model_id = self.name_to_model_id[requested_model]
            elif requested_model in self.models:
                # 用户直接使用了完整ID
                model_id = requested_model
            elif not header_model_id and not requested_model:
                # 没有指定模型，随机选择
                model_id = random.choice(list(self.models.keys()))
            else:
                # 无效的模型ID
                error_message = {
                    "error": {
                        "message": f"模型 '{requested_model or header_model_id}' 不存在",
                        "type": "invalid_request_error",
                        "code": "model_not_found",
                    }
                }
                return JSONResponse(status_code=404, content=error_message)
            
            # 替换请求中的模型为内部模型ID，方便处理
            model_request["model"] = model_id
            
            logger.info(f"使用模型ID: {model_id}")
            model_handle = self.models[model_id]
            
            # 将请求传递给相应的模型处理
            response = await model_handle.remote(model_request)
            
            # 如果是流式响应，直接返回
            if isinstance(response, StreamingResponse):
                return response
            
            # 转换响应格式，替换内部模型ID为用户友好名称
            if hasattr(response, "model") and response.model in self.model_id_to_name:
                response.model = self.model_id_to_name[response.model]
            
            # 将响应对象转换为dict
            if hasattr(response, "dict"):
                response_dict = response.dict()
                return JSONResponse(content=response_dict)
            
            return JSONResponse(content=response)
            
        except Exception as e:
            logger.exception(f"处理聊天完成请求时出错: {e}")
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
        """健康检查端点"""
        return {"status": "healthy"}


def build_app() -> serve.Application:
    """构建带有多个模型的Serve应用程序。"""

    models_handles = {}

    # 模型1: DeepSeek
    model_1_id = os.environ.get('MODEL_1_ID', "Valdemardi/DeepSeek-R1-Distill-Qwen-32B-AWQ")
    model_1_kwargs = {
        "model": model_1_id,
        "tensor_parallel_size": int(os.environ.get('MODEL_1_TENSOR_PARALLELISM', 4)),
        "quantization": os.environ.get('MODEL_1_QUANTIZE', "awq_marlin"),
        "dtype": "half",  # 使用FP16精度，提高推理速度
        "gpu_memory_utilization": 0.85,  # 控制GPU内存使用率
        "max_num_seqs": 32,  # 每次迭代的最大序列数
        "trust_remote_code": True,  # 信任远程代码，某些模型需要
    }
    models_handles[model_1_id] = VLLMDeployment.options(
        ray_actor_options={"num_cpus": 4, "num_gpus": 4}).bind(**model_1_kwargs)

    # 模型2: Mistral
    model_2_id = os.environ.get('MODEL_2_ID', "stelterlab/Mistral-Small-24B-Instruct-2501-AWQ")
    model_2_kwargs = {
        "model": model_2_id,
        "tensor_parallel_size": int(os.environ.get('MODEL_2_TENSOR_PARALLELISM', 2)),
        "quantization": os.environ.get('MODEL_2_QUANTIZE', "awq_marlin"),
        "dtype": "half",
        "gpu_memory_utilization": 0.85,
        "max_num_seqs": 32,
        "trust_remote_code": True,
    }
    models_handles[model_2_id] = VLLMDeployment.options(
        ray_actor_options={"num_cpus": 4, "num_gpus": 2}).bind(**model_2_kwargs)

    # 创建并返回多模型部署
    return MultiModelDeployment.bind(models_handles)


# 在Ray Serve中暴露的应用程序
multi_model = build_app()
