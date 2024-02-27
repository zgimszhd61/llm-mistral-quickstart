# llm-mistral

[![PyPI](https://img.shields.io/pypi/v/llm-mistral.svg)](https://pypi.org/project/llm-mistral/)
[![Changelog](https://img.shields.io/github/v/release/simonw/llm-mistral?include_prereleases&label=changelog)](https://github.com/simonw/llm-mistral/releases)
[![Tests](https://github.com/simonw/llm-mistral/workflows/Test/badge.svg)](https://github.com/simonw/llm-mistral/actions?query=workflow%3ATest)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/simonw/llm-mistral/blob/main/LICENSE)

LLM插件，通过Mistral API提供对Mistral模型的访问

## 安装

在与LLM相同的环境中安装此插件：
```bash
llm install llm-mistral
```
## 使用

首先，获取[Mistral API](https://console.mistral.ai/)的API密钥。

使用`llm keys set mistral`命令配置密钥：
```bash
llm keys set mistral
```
```
<粘贴密钥在此>
```
现在您可以访问Mistral托管的模型了。运行`llm models`获取列表。

要通过`mistral-tiny`运行提示：

```bash
llm -m mistral-tiny '一个宠物大脚的时髦名字'
```
要开始与`mistral-small`进行交互式聊天会话：
```bash
llm chat -m mistral-small
```
```
与mistral-small聊天
键入'exit'或'quit'退出
键入'!multi'以输入多行，然后键入'!end'完成
> 给宠物海象取三个自豪的名字
1. "Nanuq"，因纽特人对海象的称呼，象征着力量和韧性。
2. "Sir Tuskalot"，一个活泼而威严的名字，突出了海象独特的长牙。
3. "冰川"，一个反映海象冰冷的北极栖息地和雄伟姿态的名字。
```
要使用`mistral-medium`系统提示来解释一些代码：
```bash
cat example.py | llm -m mistral-medium -s '解释这段代码'
```

## 模型选项

所有三个模型都接受以下选项，使用`-o 名称 值`语法：

- `-o temperature 0.7`：采样温度，介于0和1之间。值越高增加随机性，较低的值更加集中和确定性。
- `-o top_p 0.1`：0.1表示仅考虑前10%概率质量中的标记。使用此选项或温度，但不要同时使用两者。
- `-o max_tokens 20`：完成中生成的最大标记数。
- `-o safe_mode 1`：打开[安全模式](https://docs.mistral.ai/platform/guardrailing/)，在模型输出中添加系统提示以添加防护栏。
- `-o random_seed 123`：设置整数随机种子以生成确定性结果。

## 嵌入

Mistral [Embeddings API](https://docs.mistral.ai/platform/client#embeddings)可用于为任何文本生成1,024维嵌入。

要嵌入单个字符串：

```bash
llm embed -m mistral-embed -c '这是文本'
```
这将返回一个包含1,024个浮点数的JSON数组。

[LLM文档](https://llm.datasette.io/en/stable/embeddings/index.html)有更多信息，包括如何批量嵌入并将结果存储在SQLite数据库中。

有关嵌入的更多信息，请参见[LLM现在提供了处理嵌入的工具](https://simonwillison.net/2023/Sep/4/llm-embeddings/)和[嵌入：它们是什么，为什么它们很重要](https://simonwillison.net/2023/Oct/23/embeddings/)。

## 开发

要在本地设置此插件，首先检出代码。然后创建一个新的虚拟环境：
```bash
cd llm-mistral
python3 -m venv venv
source venv/bin/activate
```
现在安装依赖项和测试依赖项：
```bash
llm install -e '.[test]'
```
运行测试：
```bash
pytest
```