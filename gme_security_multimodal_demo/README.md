# GME Security Multimodal Demo

一个全新独立的 Demo，用来对应“基于 GME 模型的安防多模态 RAG 知识库项目系统”这条项目线。

这个目录和 `interview_security_rag_demo` 无关，目的是把你在依图安防场景下的多模态 RAG 项目，拆成一个更容易理解和讲述的最小实现。

## Demo 目标

- 对应安防场景下的文本、图片、图文对统一检索
- 对应 GME 统一向量空间的设计思路
- 对应 Dots.OCR / 文档解析 / 章节分块 / 语义分块
- 对应 Dense + Sparse 混合检索
- 对应 Redis 短期记忆 + 长期记忆回写
- 对应 Corrective RAG + Adaptive RAG + RAGAS
- 对应 LangGraph Human-in-the-loop 工作流（已实现图节点与 `/resume`）

## 项目结构

```text
gme_security_multimodal_demo/
├── app/
│   ├── api.py
│   ├── chunker_demo.py
│   ├── embedding_demo.py
│   ├── evaluation_demo.py
│   ├── index_demo.py
│   ├── memory_demo.py
│   ├── models.py
│   ├── parser_demo.py
│   ├── run_example.py
│   ├── workflow_graph.py
│   └── workflow_demo.py
├── data/
│   └── security_multimodal_docs/
│       ├── alarm_visual_case.md
│       ├── deployment_manual.md
│       └── snapshot_qna_case.md
├── docs/
│   └── 实现说明.md
├── requirements.txt
└── 基于GME模型的安防多模态RAG知识库项目系统.txt
```

## 阅读顺序

1. 先看 `基于GME模型的安防多模态RAG知识库项目系统.txt`
2. 再看 `docs/实现说明.md`
3. 然后按下面顺序读代码：
   - `app/parser_demo.py`
   - `app/chunker_demo.py`
   - `app/embedding_demo.py`
   - `app/index_demo.py`
   - `app/memory_demo.py`
   - `app/evaluation_demo.py`
   - `app/workflow_graph.py`

## 运行方式

如果只想看结构，不需要运行。

如果想跑一个最小示例：

```bash
cd /root/autodl-tmp/rag_learning/gme_security_multimodal_demo
python -m app.run_example
```

如果想启动 API：

```bash
pip install -r requirements.txt
uvicorn app.api:app --reload --port 8012
```

## 说明

详见 `docs/改造清单.md`。GME、Milvus、Redis、Dots.OCR 仍为 Mock；LangGraph 工作流与 Any-to-Any 检索模式为真实代码路径。
