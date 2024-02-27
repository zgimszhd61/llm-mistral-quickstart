import json
import pathlib
import pytest
from pytest_httpx import IteratorStream
import llm

# 定义测试模型数据
TEST_MODELS = {
    "data": [
        {
            "id": "mistral-tiny",
            "name": "Mistral Tiny",
            "description": "一个小型模型",
        },
        {
            "id": "mistral-small",
            "name": "Mistral Small",
            "description": "一个小型模型",
        },
        {
            "id": "mistral-medium",
            "name": "Mistral Small",
            "description": "一个中型模型",
        },
        {
            "id": "mistral-large-largest",
            "name": "Mistral Large",
            "description": "一个大型模型",
        },
        {
            "id": "mistral-other",
            "name": "Mistral Other",
            "description": "另一个模型",
        },
    ]
}

# 定义会话级别的 llm 用户路径 fixture
@pytest.fixture(scope="session")
def llm_user_path(tmp_path_factory):
    tmpdir = tmp_path_factory.mktemp("llm")
    return str(tmpdir)

# 自动使用的 fixture，模拟环境
@pytest.fixture(autouse=True)
def mock_env(monkeypatch, llm_user_path):
    monkeypatch.setenv("LLM_MISTRAL_KEY", "test_key")
    monkeypatch.setenv("LLM_USER_PATH", llm_user_path)
    # 写入 mistral_models.json 文件
    (pathlib.Path(llm_user_path) / "mistral_models.json").write_text(
        json.dumps(TEST_MODELS, indent=2)
    )

# 测试缓存模型
def test_caches_models(monkeypatch, tmpdir, httpx_mock):
    httpx_mock.add_response(
        url="https://api.mistral.ai/v1/models",
        method="GET",
        json=TEST_MODELS,
    )
    llm_user_path = new_tmp_dir = str(tmpdir / "llm")
    monkeypatch.setenv("LLM_USER_PATH", llm_user_path)
    # 不应该有 llm_user_path / mistral_models.json 文件
    path = pathlib.Path(llm_user_path) / "mistral_models.json"
    assert not path.exists()
    # 列出模型应该创建该文件
    models_with_aliases = llm.get_models_with_aliases()
    assert path.exists()
    # 应该调用了该 API
    response = httpx_mock.get_request()
    assert response.url == "https://api.mistral.ai/v1/models"

# 模拟数据流的 fixture
@pytest.fixture
def mocked_stream(httpx_mock):
    httpx_mock.add_response(
        url="https://api.mistral.ai/v1/chat/completions",
        method="POST",
        stream=IteratorStream(
            [
                b'data: {"id": "cmpl-4243ee7858634455a2153d6430719956", "model": "mistral-tiny", "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": null}]}\n\n',
                b'data: {"id": "cmpl-4243ee7858634455a2153d6430719956", "object": "chat.completion.chunk", "created": 1702612156, "model": "mistral-tiny", "choices": [{"index": 0, "delta": {"role": null, "content": "我是人工智能"}, "finish_reason": null}]}\n\n',
                b'data: {"id": "cmpl-4243ee7858634455a2153d6430719956", "object": "chat.completion.chunk", "created": 1702612156, "model": "mistral-tiny", "choices": [{"index": 0, "delta": {"role": null, "content": ""}, "finish_reason": "stop"}]}\n\n',
                b"data: [DONE]",
            ]
        ),
        headers={"content-type": "text/event-stream"},
    )
    return httpx_mock

# 测试数据流
def test_stream(mocked_stream):
    model = llm.get_model("mistral-tiny")
    response = model.prompt("你好吗？")
    chunks = list(response)
    assert chunks == ["我是人工智能", ""]
    request = mocked_stream.get_request()
    assert json.loads(request.content) == {
        "model": "mistral-tiny",
        "messages": [{"role": "user", "content": "你好吗？"}],
        "temperature": 0.7,
        "top_p": 1,
        "stream": True,
    }

# 测试带选项的数据流
def test_stream_with_options(mocked_stream):
    model = llm.get_model("mistral-tiny")
    model.prompt(
        "你好吗？",
        temperature=0.5,
        top_p=0.8,
        random_seed=42,
        safe_mode=True,
        max_tokens=10,
    ).text()
    request = mocked_stream.get_request()
    assert json.loads(request.content) == {
        "model": "mistral-tiny",
        "messages": [{"role": "user", "content": "你好吗？"}],
        "temperature": 0.5,
        "top_p": 0.8,
        "random_seed": 42,
        "safe_mode": True,
        "max_tokens": 10,
        "stream": True,
    }

# 测试非数据流
def test_no_stream(httpx_mock):
    httpx_mock.add_response(
        url="https://api.mistral.ai/v1/chat/completions",
        method="POST",
        json={
            "id": "cmpl-362653b3050c4939bfa423af5f97709b",
            "object": "chat.completion",
            "created": 1702614202,
            "model": "mistral-tiny",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "我只是一个电脑程序，我没有感情。",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 16, "total_tokens": 79, "completion_tokens": 63},
        },
    )
    model = llm.get_model("mistral-tiny")
    response = model.prompt("你好吗？", stream=False)
    assert response.text() == "我只是一个电脑程序，我没有感情。"

# 自定义模型
class MistralModel(llm.AbstractModel):
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
                    # 如果未经授权：
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
