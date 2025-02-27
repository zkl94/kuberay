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

# 新增导入 transformers 库，用于 NLLB
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch

lang_code = {
    "Acehnese Arabic": "ace_Arab",
    "Acehnese Latin": "ace_Latn",
    "Mesopotamian Arabic": "acm_Arab",
    "Ta'izzi-Adeni Arabic": "acq_Arab",
    "Tunisian Arabic": "aeb_Arab",
    "Afrikaans": "afr_Latn",
    "South Levantine Arabic": "ajp_Arab",
    "Akan": "aka_Latn",
    "Amharic": "amh_Ethi",
    "North Levantine Arabic": "apc_Arab",
    "Modern Standard Arabic": "arb_Arab",
    "Modern Standard Arabic Romanized": "arb_Latn",
    "Najdi Arabic": "ars_Arab",
    "Moroccan Arabic": "ary_Arab",
    "Egyptian Arabic": "arz_Arab",
    "Assamese": "asm_Beng",
    "Asturian": "ast_Latn",
    "Awadhi": "awa_Deva",
    "Central Aymara": "ayr_Latn",
    "South Azerbaijani": "azb_Arab",
    "North Azerbaijani": "azj_Latn",
    "Bashkir": "bak_Cyrl",
    "Bambara": "bam_Latn",
    "Balinese": "ban_Latn",
    "Belarusian": "bel_Cyrl",
    "Bemba": "bem_Latn",
    "Bengali": "ben_Beng",
    "Bhojpuri": "bho_Deva",
    "Banjar Arabic": "bjn_Arab",
    "Banjar Latin": "bjn_Latn",
    "Standard Tibetan": "bod_Tibt",
    "Bosnian": "bos_Latn",
    "Buginese": "bug_Latn",
    "Bulgarian": "bul_Cyrl",
    "Catalan": "cat_Latn",
    "Cebuano": "ceb_Latn",
    "Czech": "ces_Latn",
    "Chokwe": "cjk_Latn",
    "Central Kurdish": "ckb_Arab",
    "Crimean Tatar": "crh_Latn",
    "Welsh": "cym_Latn",
    "Danish": "dan_Latn",
    "German": "deu_Latn",
    "Southwestern Dinka": "dik_Latn",
    "Dyula": "dyu_Latn",
    "Dzongkha": "dzo_Tibt",
    "Greek": "ell_Grek",
    "English": "eng_Latn",
    "Esperanto": "epo_Latn",
    "Estonian": "est_Latn",
    "Basque": "eus_Latn",
    "Ewe": "ewe_Latn",
    "Faroese": "fao_Latn",
    "Fijian": "fij_Latn",
    "Finnish": "fin_Latn",
    "Fon": "fon_Latn",
    "French": "fra_Latn",
    "Friulian": "fur_Latn",
    "Nigerian Fulfulde": "fuv_Latn",
    "Scottish Gaelic": "gla_Latn",
    "Irish": "gle_Latn",
    "Galician": "glg_Latn",
    "Guarani": "grn_Latn",
    "Gujarati": "guj_Gujr",
    "Haitian Creole": "hat_Latn",
    "Hausa": "hau_Latn",
    "Hebrew": "heb_Hebr",
    "Hindi": "hin_Deva",
    "Chhattisgarhi": "hne_Deva",
    "Croatian": "hrv_Latn",
    "Hungarian": "hun_Latn",
    "Armenian": "hye_Armn",
    "Igbo": "ibo_Latn",
    "Ilocano": "ilo_Latn",
    "Indonesian": "ind_Latn",
    "Icelandic": "isl_Latn",
    "Italian": "ita_Latn",
    "Javanese": "jav_Latn",
    "Japanese": "jpn_Jpan",
    "Kabyle": "kab_Latn",
    "Jingpho": "kac_Latn",
    "Kamba": "kam_Latn",
    "Kannada": "kan_Knda",
    "Kashmiri Arabic": "kas_Arab",
    "Kashmiri Devanagari": "kas_Deva",
    "Georgian": "kat_Geor",
    "Central Kanuri Arabic": "knc_Arab",
    "Central Kanuri Latin": "knc_Latn",
    "Kazakh": "kaz_Cyrl",
    "Kabiyè": "kbp_Latn",
    "Kabuverdianu": "kea_Latn",
    "Khmer": "khm_Khmr",
    "Kikuyu": "kik_Latn",
    "Kinyarwanda": "kin_Latn",
    "Kyrgyz": "kir_Cyrl",
    "Kimbundu": "kmb_Latn",
    "Northern Kurdish": "kmr_Latn",
    "Kikongo": "kon_Latn",
    "Korean": "kor_Hang",
    "Lao": "lao_Laoo",
    "Ligurian": "lij_Latn",
    "Limburgish": "lim_Latn",
    "Lingala": "lin_Latn",
    "Lithuanian": "lit_Latn",
    "Lombard": "lmo_Latn",
    "Latgalian": "ltg_Latn",
    "Luxembourgish": "ltz_Latn",
    "Luba-Kasai": "lua_Latn",
    "Ganda": "lug_Latn",
    "Luo": "luo_Latn",
    "Mizo": "lus_Latn",
    "Standard Latvian": "lvs_Latn",
    "Magahi": "mag_Deva",
    "Maithili": "mai_Deva",
    "Malayalam": "mal_Mlym",
    "Marathi": "mar_Deva",
    "Minangkabau Arabic ": "min_Arab",
    "Minangkabau Latin": "min_Latn",
    "Macedonian": "mkd_Cyrl",
    "Plateau Malagasy": "plt_Latn",
    "Maltese": "mlt_Latn",
    "Meitei Bengali": "mni_Beng",
    "Halh Mongolian": "khk_Cyrl",
    "Mossi": "mos_Latn",
    "Maori": "mri_Latn",
    "Burmese": "mya_Mymr",
    "Dutch": "nld_Latn",
    "Norwegian Nynorsk": "nno_Latn",
    "Norwegian Bokmål": "nob_Latn",
    "Nepali": "npi_Deva",
    "Northern Sotho": "nso_Latn",
    "Nuer": "nus_Latn",
    "Nyanja": "nya_Latn",
    "Occitan": "oci_Latn",
    "West Central Oromo": "gaz_Latn",
    "Odia": "ory_Orya",
    "Pangasinan": "pag_Latn",
    "Eastern Panjabi": "pan_Guru",
    "Papiamento": "pap_Latn",
    "Western Persian": "pes_Arab",
    "Polish": "pol_Latn",
    "Portuguese": "por_Latn",
    "Dari": "prs_Arab",
    "Southern Pashto": "pbt_Arab",
    "Ayacucho Quechua": "quy_Latn",
    "Romanian": "ron_Latn",
    "Rundi": "run_Latn",
    "Russian": "rus_Cyrl",
    "Sango": "sag_Latn",
    "Sanskrit": "san_Deva",
    "Santali": "sat_Olck",
    "Sicilian": "scn_Latn",
    "Shan": "shn_Mymr",
    "Sinhala": "sin_Sinh",
    "Slovak": "slk_Latn",
    "Slovenian": "slv_Latn",
    "Samoan": "smo_Latn",
    "Shona": "sna_Latn",
    "Sindhi": "snd_Arab",
    "Somali": "som_Latn",
    "Southern Sotho": "sot_Latn",
    "Spanish": "spa_Latn",
    "Tosk Albanian": "als_Latn",
    "Sardinian": "srd_Latn",
    "Serbian": "srp_Cyrl",
    "Swati": "ssw_Latn",
    "Sundanese": "sun_Latn",
    "Swedish": "swe_Latn",
    "Swahili": "swh_Latn",
    "Silesian": "szl_Latn",
    "Tamil": "tam_Taml",
    "Tatar": "tat_Cyrl",
    "Telugu": "tel_Telu",
    "Tajik": "tgk_Cyrl",
    "Tagalog": "tgl_Latn",
    "Thai": "tha_Thai",
    "Tigrinya": "tir_Ethi",
    "Tamasheq Latin": "taq_Latn",
    "Tamasheq Tifinagh": "taq_Tfng",
    "Tok Pisin": "tpi_Latn",
    "Tswana": "tsn_Latn",
    "Tsonga": "tso_Latn",
    "Turkmen": "tuk_Latn",
    "Tumbuka": "tum_Latn",
    "Turkish": "tur_Latn",
    "Twi": "twi_Latn",
    "Central Atlas Tamazight": "tzm_Tfng",
    "Uyghur": "uig_Arab",
    "Ukrainian": "ukr_Cyrl",
    "Umbundu": "umb_Latn",
    "Urdu": "urd_Arab",
    "Northern Uzbek": "uzn_Latn",
    "Venetian": "vec_Latn",
    "Vietnamese": "vie_Latn",
    "Waray": "war_Latn",
    "Wolof": "wol_Latn",
    "Xhosa": "xho_Latn",
    "Eastern Yiddish": "ydd_Hebr",
    "Yoruba": "yor_Latn",
    "Yue Chinese": "yue_Hant",
    "Chinese Simplified": "zho_Hans",
    "Chinese Traditional": "zho_Hant",
    "Standard Malay": "zsm_Latn",
    "Zulu": "zul_Latn",
}

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

    async def generate_streaming(self, prompt: str, sampling_params: dict, request_id: str = None) -> AsyncGenerator:
        """Generate streaming response directly - to be called from other deployments"""
        if request_id is None:
            request_id = random_uuid()

        # Convert dict to SamplingParams if needed
        if isinstance(sampling_params, dict):
            sampling_params = SamplingParams(**sampling_params)

        # Get generation results
        results_generator = self.engine.generate(
            prompt, sampling_params, request_id)

        # First chunk with assistant role
        chunk = {
            "id": random_uuid(),
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": self.model_id,
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant"},
                    "finish_reason": None
                }
            ]
        }
        yield f"data: {json.dumps(chunk)}\n\n"

        # Track previously sent text to compute deltas
        previous_text = ""

        # Stream content chunks
        async for request_output in results_generator:
            if len(request_output.outputs) == 0:
                continue

            # Get the full text generated so far
            current_text = request_output.outputs[0].text

            # Compute the delta (only the new part)
            if current_text.startswith(previous_text):
                delta_text = current_text[len(previous_text):]
            else:
                # Fallback in case of text replacement (unlikely but possible)
                delta_text = current_text

            # Skip empty deltas
            if not delta_text:
                continue

            # Update previous text for next iteration
            previous_text = current_text

            # Create the chunk with only the delta text
            chunk = {
                "id": random_uuid(),
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": self.model_id,
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
            "model": self.model_id,
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

    async def handle_chat_request(self, request: dict, model_id: str) -> Union[dict, dict]:
        """Process a chat completion request"""
        request_id = random_uuid()

        # Extract messages from request
        messages = request.get("messages", [])
        if not messages:
            raise ValueError("Messages array cannot be empty")

        # Convert messages to prompt text
        prompt = messages_to_prompt(messages, model_id)

        # Extract sampling parameters
        sampling_params = {
            "temperature": request.get("temperature", 1.0),
            "top_p": request.get("top_p", 1.0),
            "max_tokens": request.get("max_tokens", 1024),
            "stop": request.get("stop"),
            "frequency_penalty": request.get("frequency_penalty", 0.0),
            "presence_penalty": request.get("presence_penalty", 0.0),
        }

        # Check if streaming is requested
        stream = request.get("stream", False)

        if stream:
            # For streaming requests, return information necessary to create a streaming response
            return {
                "stream": True,
                "prompt": prompt,
                "sampling_params": sampling_params,
                "model_id": model_id
            }

        # For non-streaming, use the existing code
        # Convert dict to SamplingParams
        sampling_params_obj = SamplingParams(**sampling_params)

        # Get generation results
        results_generator = self.engine.generate(
            prompt, sampling_params_obj, request_id)

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

            # If we got back a streaming info dictionary
            if isinstance(response, dict) and response.get("stream") is True:
                # Return the dictionary with streaming info
                return response

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


# 新增 NLLB 翻译模型部署
@serve.deployment(name="NLLBDeployment")
class NLLBDeployment:
    def __init__(self, model_id="facebook/nllb-200-3.3B"):
        """初始化NLLB翻译模型"""
        self.model_id = model_id
        # Log model loading
        logger.info(f"开始加载NLLB模型: {model_id}")
        start_time = time.time()

        # Load NLLB model and tokenizer
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_id, torch_dtype=torch.float16, attn_implementation="flash_attention_2")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

        # If CUDA is available, move model to GPU
        if torch.cuda.is_available():
            self.model = self.model.to("cuda")

        # Log model load time
        logger.info(
            f"NLLB model load time: {time.time() - start_time:.2f} seconds")

    async def translate(self, text: str, target_lang: str) -> str:
        """Use NLLB model to translate text"""
        logger.info(f'target_lang: {target_lang}')

        if target_lang not in lang_code.values():
            logger.warning(
                f"Target language is not valid: {target_lang}, using zho_Hans by default")
            target_lang = "zho_Hans"

        device = "cuda" if torch.cuda.is_available() else "cpu"

        inputs = self.tokenizer(text, return_tensors="pt").to(device)
        logger.info(f"Input tokens: {inputs}")

        translated_tokens = self.model.generate(
            **inputs,
            forced_bos_token_id=self.tokenizer.convert_tokens_to_ids(
                target_lang),
            max_length=512
        )
        logger.info(f"Translated tokens: {translated_tokens}")

        # Decode translated tokens
        translation = self.tokenizer.batch_decode(
            translated_tokens, skip_special_tokens=True)[0]
        logger.info(f"Translation: {translation}")

        return translation

    async def handle_translation_request(self, request: dict) -> dict:
        """处理翻译请求"""
        # Get text and target language from request
        text = request.get("text", "")
        target_lang = request.get("target_lang", "zho_Hans")

        if not text:
            return {
                "error": {
                    "message": "Text cannot be empty",
                    "type": "invalid_request_error",
                    "code": 400
                }
            }

        try:
            # 翻译文本
            translation = await self.translate(text, target_lang)

            # 返回翻译结果
            return {
                "id": random_uuid(),
                "object": "translation",
                "created": int(time.time()),
                "model": self.model_id,
                "translation": translation,
                "target_lang": target_lang
            }

        except Exception as e:
            logger.exception(f"翻译错误: {e}")
            return {
                "error": {
                    "message": str(e),
                    "type": "translation_error",
                    "code": 500
                }
            }

    async def __call__(self, request_dict: dict) -> Any:
        """处理API请求"""
        try:
            response = await self.handle_translation_request(request_dict)
            return JSONResponse(content=response)

        except Exception as e:
            logger.exception(f"处理请求错误: {e}")
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
    def __init__(self, models: Dict[str, DeploymentHandle], nllb_model: Optional[DeploymentHandle] = None):
        self.models = models
        self.nllb_model = nllb_model
        # Create mapping from model IDs to friendly names
        self.model_id_to_name = {
            "Valdemardi/DeepSeek-R1-Distill-Llama-70B-AWQ": "deepseek-r1-70b",
            "stelterlab/Mistral-Small-24B-Instruct-2501-AWQ": "mistral-small-24b",
            "facebook/nllb-200-3.3B": "nllb-3.3b",  # 添加NLLB模型
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

        # 如果NLLB模型可用，也添加到列表中
        if self.nllb_model:
            available_models.append({
                "id": "nllb-3.3b",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "kuberay-user",
                "capabilities": {
                    "translation": True
                }
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
            response = await model_handle.handle_chat_request.remote(model_request, model_id)

            # Check if this is a streaming response info dictionary
            if isinstance(response, dict) and response.get("stream") is True:
                # Create a streaming response
                # Use handle.options(stream=True) to enable streaming
                stream_generator = model_handle.options(stream=True).generate_streaming.remote(
                    response.get("prompt"),
                    response.get("sampling_params")
                )

                # Return a streaming response
                return StreamingResponse(
                    stream_generator,
                    media_type="text/event-stream"
                )

            # For non-streaming responses, just pass through
            return JSONResponse(content=response)

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

    # 新增翻译API端点
    @app.post("/v1/translations")
    async def create_translation(self, request: Request):
        """创建翻译，使用NLLB模型"""
        if not self.nllb_model:
            error_message = {
                "error": {
                    "message": "Translation service is not available",
                    "type": "service_unavailable",
                    "code": "translation_unavailable",
                }
            }
            return JSONResponse(status_code=503, content=error_message)

        try:
            translation_request = await request.json()

            # 调用NLLB模型进行翻译
            response = await self.nllb_model.handle_translation_request.remote(translation_request)

            # 返回翻译结果
            return JSONResponse(content=response)

        except Exception as e:
            logger.exception(f"Error handling translation request: {e}")
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

    # 创建NLLB模型实例
    nllb_model = NLLBDeployment.options(
        ray_actor_options={"num_cpus": 2, "num_gpus": 2}).bind()

    # Create and return multi-model deployment with NLLB
    return MultiModelDeployment.bind(models_handles, nllb_model)


# Application exposed in Ray Serve
multi_model = build_app()
