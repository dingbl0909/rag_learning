# Security Agent Demo

这是一个面向安防场景的简化版多 Agent 学习项目，目标是把 `ctrip_assistant` 里最值得学习的 Agent 编排思路抽出来，做成更小、更容易理解、更容易运行的版本。

## 项目目标

这个 demo 重点保留了以下能力：

- `LangChain` 的 `prompt | llm.bind_tools(...)`
- `LangGraph` 的状态管理与条件路由
- 主 Agent + 子 Agent 的委派模式
- `dialog_state` 状态栈
- `safe_tools` / `sensitive_tools` 分层
- `interrupt_before` 敏感操作确认
- FastAPI 最小接口

## 目录结构

```text
security_agent_demo/
├── app.py
├── config.py
├── llm.py
├── state.py
├── models.py
├── tools.py
├── agents.py
├── graph.py
├── schemas.py
├── requirements.txt
└── demo_data/
    ├── users.json
    ├── cameras.json
    ├── video_events.json
    ├── snapshots.json
    └── knowledge_base.json
```

## 业务拆分

项目保留了三个子 Agent：

- `video_agent`
  - 视频流告警
  - 录像片段定位
  - 摄像头状态查询
- `image_agent`
  - 抓拍记录查询
  - 图片分析说明
  - 图片取证问答
- `ops_agent`
  - SOP 检索
  - 运维知识问答
  - 工单建议

主 Agent 负责统一意图识别和任务分发。

## 和原项目的映射关系

如果你已经看过 `ctrip_assistant`，可以这样对照：

- `security_agent_demo/graph.py`
  - 对应 `ctrip_assistant/graph_chat/finally_graph.py`
- `security_agent_demo/state.py`
  - 对应 `ctrip_assistant/graph_chat/state.py`
- `security_agent_demo/agents.py`
  - 对应 `ctrip_assistant/graph_chat/assistant.py` + `graph_chat/agent_assistant.py`
- `security_agent_demo/models.py`
  - 对应 `ctrip_assistant/graph_chat/base_data_model.py`
- `security_agent_demo/tools.py`
  - 对应 `ctrip_assistant/tools/*.py` + `tools/tools_handler.py`

## 关键设计

### 1. 主 Agent 的委派

主 Agent 不直接处理所有业务，而是通过结构化工具把任务委派给子 Agent。

例如：

- `ToVideoAgent`
- `ToImageAgent`
- `ToOpsAgent`

### 2. 状态栈

`state.py` 中定义了 `dialog_state`，用来记录当前处于哪个子流程。

- 进入子 Agent 时压栈
- 退出子 Agent 时弹栈

### 3. 工具分层

每个子 Agent 的工具被分成：

- `safe_tools`：查询类、低风险
- `sensitive_tools`：执行类、高风险

### 4. 敏感操作确认

在 `graph.py` 中，图使用了 `interrupt_before`：

- `video_sensitive_tools`
- `ops_sensitive_tools`

这表示执行敏感操作前先暂停，等待用户确认。

## 模型接入

默认接入已经部署好的 `Qwen3-8B` OpenAI 兼容接口：

- `http://127.0.0.1:6006/v1`

配置在 `config.py` 和 `llm.py` 中。

## 运行方式

先安装依赖：

```bash
pip install -r security_agent_demo/requirements.txt
```

再启动服务：

```bash
uvicorn security_agent_demo.app:app --host 0.0.0.0 --port 8010
```

## 接口说明

### 1. 发起对话

`POST /chat`

示例：

```bash
curl -X POST http://127.0.0.1:8010/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "仓库北门摄像头为什么离线？给我排查建议",
    "thread_id": "demo-thread-1",
    "user_id": "ops_001"
  }'
```

### 2. 继续敏感操作

`POST /chat/continue`

示例：

```bash
curl -X POST http://127.0.0.1:8010/chat/continue \
  -H "Content-Type: application/json" \
  -d '{
    "thread_id": "demo-thread-1",
    "user_id": "ops_001",
    "approve": true
  }'
```

## 推荐学习顺序

1. 先看 `models.py`
   - 理解主 Agent 委派工具和子 Agent 退出工具
2. 再看 `state.py`
   - 理解 `dialog_state` 为什么是栈
3. 再看 `tools.py`
   - 理解工具是怎么抽象的
4. 再看 `agents.py`
   - 理解 prompt、tool binding、主/子 Agent 的职责
5. 最后看 `graph.py`
   - 理解总图如何把这些拼起来

## 适合你重点观察的内容

如果你想把这套思路迁移到自己的安防项目，优先关注：

- 主 Agent 如何分发到视频流 / 图片流 / 运维场景
- `safe_tools` 和 `sensitive_tools` 如何设计
- `interrupt_before` 如何做人工确认
- 如何用 mock 工具先把流程跑通，再替换成真实后端接口
