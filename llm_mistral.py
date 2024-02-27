import click
from httpx_sse import connect_sse
import httpx
import json
import llm
from pydantic import Field
from typing import Optional

# 默认别名
DEFAULT_ALIASES = {
    "mistral/mistral-tiny": "mistral-tiny",
    "mistral/mistral-small": "mistral-small",
    "mistral/mistral-medium": "mistral-medium",
    "mistral/mistral-large-latest": "mistral-large",
}

# 注册模型
@llm.hookimpl
def register_models(register):
    for model_id in get_model_ids():
        our_model_id = "mistral/" + model_id
        alias = DEFAULT_ALIASES.get(our_model_id)
        aliases = [alias] if alias else []
        register(Mistral(our_model_id, model_id), aliases=aliases)

# 注册嵌入模型
@llm.hookimpl
def register_embedding_models(register):
    register(MistralEmbed())

# 刷新模型列表
def refresh_models():
    user_dir = llm.user_dir()
    mistral_models = user_dir / "mistral_models.json"
    key = llm.get_key("", "mistral", "LLM_MISTRAL_KEY")
    if not key:
        raise click.ClickException(
            "你必须设置 'mistral' 键或 LLM_MISTRAL_KEY 环境变量。"
        )
    response = httpx.get(
        "https://api.mistral.ai/v1/models", headers={"Authorization": f"Bearer {key}"}
    )
    response.raise_for_status()
    models = response.json()
    mistral_models.write_text(json.dumps(models, indent=2))
    return models

# 获取模型ID
def get_model_ids():
    user_dir = llm.user_dir()
    mistral_models = user_dir / "mistral_models.json"
    if mistral_models.exists():
        models = json.loads(mistral_models.read_text())
    else:
        models = refresh_models()
    return [model["id"] for model in models["data"] if "embed" not in model["id"]]

# 注册命令
@llm.hookimpl
def register_commands(cli):
    @cli.group()
    def mistral():
        "与 llm-mistral 插件相关的命令"

    @mistral.command()
    def refresh():
        "刷新可用的 Mistral 模型列表"
        before = set(get_model_ids())
        refresh_models()
        after = set(get_model_ids())
        added = after - before
        removed = before - after
        if added:
            click.echo(f"添加的模型: {', '.join(added)}", err=True)
        if removed:
            click.echo(f"移除的模型: {', '.join(removed)}", err=True)
        if added or removed:
            click.echo("新的模型列表:", err=True)
            for model_id in get_model_ids():
                click.echo(model_id, err=True)
        else:
            click.echo("没有变化", err=True)

# Mistral 类
class Mistral(llm.Model):
    can_stream = True

    class Options(llm.Options):
        temperature: Optional[float] = Field(
            description=(
                "决定采样温度。较高的值（如0.8）增加了随机性，"
                "而较低的值（如0.2）使输出更加集中和确定性。"
            ),
            ge=0,
            le=1,
            default=0.7,
        )
        top_p: Optional[float] = Field(
            description=(
                "核心采样，模型考虑 top_p 概率质量的令牌。"
                "例如，0.1 表示仅考虑前10％的概率质量中的令牌。"
            ),
            ge=0,
            le=1,
            default=1,
        )
        max_tokens: Optional[int] = Field(
            description="生成完成时的最大令牌数。",
            ge=0,
            default=None,
        )
        safe_mode: Optional[bool] = Field(
            description="是否在所有对话之前注入安全提示。",
            default=False,
        )
        random_seed: Optional[int] = Field(
            description="设置随机抽样的种子以生成确定性结果。",
            default=None,
        )

    def __init__(self, our_model_id, mistral_model_id):
        self.model_id = our_model_id
        self.mistral_model_id = mistral_model_id

    def build_messages(self, prompt, conversation):
        messages = []
        if not conversation:
            if prompt.system:
                messages.append({"role": "system", "content": prompt.system})
            messages.append({"role": "user", "content": prompt.prompt})
            return messages
        current_system = None
        for prev_response in conversation.responses:
            if (
                prev_response.prompt.system
                and prev_response.prompt.system != current_system
            ):
                messages.append(
                    {"role": "system", "content": prev_response.prompt.system}
                )
                current_system = prev_response.prompt.system
            messages.append({"role": "user", "content": prev_response.prompt.prompt})
            messages.append({"role": "assistant", "content": prev_response.text()})
        if prompt.system and prompt.system != current_system:
            messages.append({"role": "system", "content": prompt.system})
        messages.append({"role": "user", "content": prompt.prompt})
        return messages

    def execute(self, prompt, stream, response, conversation):
        key = llm.get_key("", "mistral", "LLM_MISTRAL_KEY") or getattr(self, "key", None)
        messages = self.build_messages(prompt, conversation)
        response._prompt_json = {"messages": messages}
        body = {
            "model": self.mistral_model_id,
            "messages": messages,
        }
        if prompt.options.temperature:
            body["temperature"] = prompt.options.temperature
        if prompt.options.top_p:
            body["top_p"] = prompt.options.top_p
        if prompt.options.max_tokens:
            body["max_tokens"] = prompt.options.max_tokens
        if prompt.options.safe_mode:
            body["safe_mode"] = prompt.options.safe_mode
        if prompt.options.random_seed:
            body["random_seed"] = prompt.options.random_seed
        if stream:
            body["stream"] = True
            with httpx.Client() as client:
                with connect_sse(
                    client,
                    "POST",
                    "https://api.mistral.ai/v1/chat/completions",
                    headers={
                        "Content-Type": "application/json",
                        "Accept": "application/json",
                        "Authorization": f"Bearer {key}",
                    },
                    json=body,
                    timeout=None,
                ) as event_source:
                    # 未授权的情况下:
                    event_source.response.raise_for_status()
                    for sse in event_source.iter_sse():
                        if sse.data != "[DONE]":
                            try:
                                yield sse.json()["choices"][0]["delta"]["content"]
                            except KeyError:
                                pass
        else:
            with httpx.Client() as client:
                api_response = client.post(
                    "https://api.mistral.ai/v1/chat/completions",
                    headers={
                        "Content-Type": "application/json",
                        "Accept": "application/json",
                        "Authorization": f"Bearer {key}",
                    },
                    json=body,
                    timeout=None,
                )
                api_response.raise_for_status()
                yield api_response.json()["choices"][0]["message"]["content"]
                response.response_json = api_response.json()

# MistralEmbed 类
class MistralEmbed(llm.EmbeddingModel):
    model_id = "mistral-embed"
    batch_size = 10

    def embed_batch(self, texts):
        key = llm.get_key("", "mistral", "LLM_MISTRAL_KEY")
        with httpx.Client() as client:
            api_response = client.post(
                "https://api.mistral.ai/v1/embeddings",
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                    "Authorization": f"Bearer {key}",
                },
                json={
                    "model": "mistral-embed",
                    "input": list(texts),
                    "encoding_format": "float",
                },
                timeout=None,
            )
            api_response.raise_for_status()
            return [item["embedding"] for item in api_response.json()["data"]]
