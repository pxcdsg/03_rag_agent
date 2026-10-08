# 03_rag_agent

一个循序渐进的 RAG 项目。当前完成文档读取、文本切块、基础相似度检索，以及基于检索结果的模型回答。

## 当前功能

- 读取 `documents/` 下的 `.md` 和 `.txt` 文件
- 把长文本切成长度固定的文本块
- 相邻文本块保留一部分重叠，避免上下文被切断
- 为每块记录来源文件和块编号
- 把问题和文本块转换为字符词频特征
- 使用余弦相似度找出最相关的文本块
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

## 运行

```powershell
python main.py
```

## 后续阶段

1. 用 Embedding 替换字符词频特征
2. 增加多轮对话
3. 增加 FastAPI 或 Streamlit 界面
