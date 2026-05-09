# Interview Security RAG Demo

一个适合在面试中展示的安防知识问答 Demo。

它不是对 `Multimodal_RAG` 的完整复制，而是从现有代码和文档中抽取出最能体现项目价值的部分，重新组织成一个更容易讲清楚、也更容易跑起来的最小可运行项目。

现在这个仓库包含两层内容：

- 可直接运行的轻量 Demo 主链路
- 与 `智能安防知识问答平台.txt` 对齐的完整架构模块
- 与 `基于GME模型的安防多模态RAG知识库项目系统.txt` 对齐的多模态安防 Demo 模块

## Demo 目标

- 展示安防垂直知识库问答场景
- 展示章节分块 + 语义补充分块
- 展示 Dense/Sparse 混合检索思想
- 展示基于工作流的动态路由
- 展示低置信度结果的人审中断
- 展示可解释回答和证据返回

## 与原始项目的对应关系

这个 Demo 吸收了两个来源的思路：

- `Multimodal_RAG`
  - 文档切分
  - 混合检索
  - 工作流式问答
  - 低分结果的人审
- `智能安防知识问答平台.txt`
  - 安防业务背景
  - 设备接入 / 布控告警 / 部署排障等知识域
  - 面向研发、实施、运维团队的问答目标

## 项目结构

```text
interview_security_rag_demo/
├── app/
│   ├── api.py
│   ├── architecture_overview.py
│   ├── chunking.py
│   ├── config.py
│   ├── embeddings/
│   │   ├── bge_encoder.py
│   │   └── sparse_encoder.py
│   ├── evaluation/
│   │   ├── adaptive_rag.py
│   │   ├── corrective_rag.py
│   │   └── ragas_loop.py
│   ├── graph/
│   │   ├── builder.py
│   │   ├── nodes.py
│   │   └── state.py
│   ├── ingestion/
│   │   ├── chunk_pipeline.py
│   │   └── parsers.py
│   ├── knowledge_base.py
│   ├── models.py
│   ├── multimodal_security/
│   │   ├── chunker.py
│   │   ├── demo_runner.py
│   │   ├── embedding.py
│   │   ├── evaluation.py
│   │   ├── index.py
│   │   ├── memory.py
│   │   ├── models.py
│   │   ├── parser.py
│   │   └── workflow.py
│   ├── retrieval.py
│   ├── vectorstore/
│   │   ├── milvus_cluster.py
│   │   └── milvus_schema.py
│   └── workflow.py
├── data/
│   ├── multimodal_security_docs/
│   │   ├── device_deployment_manual.md
│   │   ├── frontline_snapshot_case.md
│   │   └── video_alarm_case.md
│   └── security_docs/
│       ├── alarm_response.md
│       ├── deployment_troubleshooting.md
│       └── device_access.md
├── deploy/
│   └── docker-compose.milvus.yml
├── docs/
│   └── 架构补充说明.md
├── requirements.txt
├── requirements-extended.txt
├── 基于GME模型的安防多模态RAG知识库项目系统.txt
├── 智能安防知识问答平台.txt
├── 面试讲稿.md
└── README.md
```

## 模块说明

### 1. 可运行主链路

- `app/api.py`
- `app/knowledge_base.py`
- `app/retrieval.py`
- `app/workflow.py`

这部分可以直接启动并演示问答、证据返回和人工审核。

### 2. 与 txt 对齐的架构补充模块

- `app/ingestion/`
  - 对应 `PyMuPDF`、`UnstructuredLoader`、`Tesseract OCR`
- `app/embeddings/`
  - 对应 `bge-large-zh-v1.5` 和 `BM25 / DAAT`
- `app/vectorstore/`
  - 对应 `Milvus Dense/Sparse 双索引` 和集群设计
- `app/graph/`
  - 对应 `LangGraph` 动态工作流
- `app/evaluation/`
  - 对应 `Corrective RAG`、`Adaptive RAG`、`RAGAS`

这些模块允许你把项目讲成“完整方案”，即使当前不要求全部真实运行。

### 3. 多模态安防 RAG 理解型 Demo

- `基于GME模型的安防多模态RAG知识库项目系统.txt`
  - 这是结合依图安防工作内容重写后的项目文档版本
- `app/multimodal_security/`
  - 这是根据该文档整理出的理解型 demo 实现
- `data/multimodal_security_docs/`
  - 这里放了视频告警、设备部署、现场抓拍三类安防样例资料

这部分重点不是页面展示，而是帮助你从代码层面理解：

- OCR / 解析结果如何进入知识单元
- GME 统一向量空间在代码里如何抽象
- Dense / Sparse 混合检索如何组织
- Redis 短期记忆和长期上下文如何配合
- Corrective / Adaptive / RAGAS 如何串到工作流里

## 运行方式

1. 安装依赖

```bash
pip install -r requirements.txt
```

如果你想把“完整架构模块”的依赖也一起装上，可以参考：

```bash
pip install -r requirements-extended.txt
```

2. 启动服务

```bash
uvicorn app.api:app --reload --port 8011
```

3. 访问接口

- `GET /`：查看服务状态
- `GET /architecture`：查看与 txt 对齐的整体架构说明
- `GET /architecture/langgraph`：查看 LangGraph 蓝图
- `GET /architecture/milvus`：查看 Milvus 集群与集合设计
- `GET /multimodal-security/example`：查看多模态安防 RAG demo 示例输出
- `POST /rebuild-index`：重建知识库索引
- `POST /chat`：发起问答
- `POST /chat/continue`：对低置信度回答进行人工确认

## 与 txt 的匹配度

这个 Demo 现在已经不是“只保留思想”的版本，而是补到了“主链路可跑，扩展架构齐全”的状态。

### 已显式补齐的 txt 能力点

- Milvus 集群与 Docker 部署蓝图
- PyMuPDF / Unstructured / Tesseract 解析模块
- 章节段落分块 + SemanticChunker 模块
- bge-large-zh-v1.5 稠密向量模块
- BM25 + DAAT 稀疏检索模块
- Dense/Sparse 双索引 schema
- LangGraph 风格动态工作流模块
- Corrective RAG / Adaptive RAG 模块
- RAGAS 评估闭环模块

### 当前仍然是 blueprint 的部分

- 外部模型服务调用
- 真正的 Milvus 集群运行
- 真正的 PDF / OCR 解析执行
- 真正的 LangGraph 编译执行
- 真正的 RAGAS 指标调用

这些模块目前的目标是“结构和接口到位，便于讲清楚架构与演进路径”。

## 示例问题

- `摄像头接入平台后一直离线，应该怎么排查？`
- `布控告警频繁误报时，通常先检查哪些配置？`
- `Milvus 容器启动后检索为空，排查顺序是什么？`
- `我要直接升级告警规则并重启服务，请给出建议。`

最后一个问题会更容易触发低置信度或风险审查流程，适合现场演示 `human-in-the-loop`。

## 面试讲解建议

你可以按下面这个顺序介绍：

1. 先讲业务目标：安防知识分散，传统关键词搜索不够用。
2. 再讲链路：文档入库 -> 分块 -> 混合检索 -> 工作流回答 -> 人工审核。
3. 然后讲工程化模块：
   - `ingestion` 对应文档解析
   - `vectorstore` 对应 Milvus 设计
   - `embeddings` 对应 BGE / BM25
   - `graph` 对应 LangGraph 工作流
   - `evaluation` 对应 Corrective / Adaptive / RAGAS
4. 最后强调这个 Demo 既能跑主流程，也能展示完整架构拆分，不会只停留在概念描述。
