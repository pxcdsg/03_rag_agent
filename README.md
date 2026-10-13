# 03_rag_agent

一个循序渐进的 RAG 项目。当前完成文档读取、段落切块、字符词频检索、Embedding 语义检索，以及基于检索结果的模型回答。

## 当前功能

- 读取 `documents/` 下的 `.md` 和 `.txt` 文件
- 优先按空行和段落切分文本
- 只有超长段落才使用字符重叠切分
- 为每块记录来源文件和块编号
- 把问题和文本块转换为字符词频特征
- 使用余弦相似度找出最相关的文本块
- 使用 Ollama Embedding 进行语义检索
- 缓存文档向量，问题向量每次实时计算
- 把相关文本块和问题一起交给 DeepSeek 或 Ollama
- 要求模型根据参考资料回答并标注来源

## 安装

```powershell
python -m pip install -r requirements.txt
```

## 配置

将 `.env.example` 复制为 `.env`，然后填写配置：

```text
LLM_PROVIDER=deepseek

DEEPSEEK_API_KEY=你的密钥
DEEPSEEK_MODEL=deepseek-chat

OLLAMA_BASE_URL=http://localhost:11434/v1
OLLAMA_MODEL=gemma3:4b
```

使用 Ollama 时，把 `LLM_PROVIDER` 改成 `ollama`。

语义检索同样使用本地 Ollama，需要先下载 Embedding 模型：

```powershell
ollama pull embeddinggemma
```

`RETRIEVAL_MODE` 可以选择：

```text
embedding：使用 Embedding 语义检索
lexical：使用原来的字符词频检索
```

## 运行

```powershell
python main.py
```

## 检索评测

运行独立的检索评测集，比较字面检索和 Embedding 检索：

```powershell
python evaluate_retrieval.py
```

评测语料位于 `evaluations/documents/`，问题位于
`evaluations/retrieval_cases.json`。脚本会输出两种方式的 Top-1
和 Top-3 命中率。

## 后续阶段

1. 增加多轮对话
2. 增加 FastAPI 或 Streamlit 界面
3. 用向量数据库替换本地 JSON 缓存
