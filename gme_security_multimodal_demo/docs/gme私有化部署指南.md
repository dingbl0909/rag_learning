# gme-Qwen2-VL-7B-Instruct 私有化部署指南

本指南按“最小生产单元”部署：GME 负责多模态向量，Milvus 负责知识库向量索引和长期记忆，Redis 负责短期会话和查询缓存。

## 1. 准备 Python 环境

建议使用 NVIDIA GPU 机器，显存至少 16GB，生产环境建议 24GB 以上。

```bash
conda create -n gme-rag python=3.10 -y
conda activate gme-rag
cd /root/rag_learning/gme_security_multimodal_demo
python -m pip install -r requirements.txt
```

如果需要固定 CUDA 版本，请先按服务器驱动安装对应的 PyTorch，再安装 `requirements.txt` 中其余依赖。

## 2. 下载 GME 模型

联网机器上下载模型：

```bash
python -m pip install modelscope
modelscope download \
  --model iic/gme-Qwen2-VL-7B-Instruct \
  --local_dir /models/gme-Qwen2-VL-7B-Instruct
```

将 `/models/gme-Qwen2-VL-7B-Instruct` 同步到私有化服务器。服务启动时不要依赖公网模型下载。

## 3. 启动 Milvus 和 Redis

项目根目录已经提供 `docker-compose.yml`，包含：

- Milvus Standalone：向量索引服务，端口 `19530`
- Redis：短期记忆和查询缓存，端口 `6379`
- etcd：Milvus 元数据依赖
- MinIO：Milvus 对象存储依赖，控制台端口 `9001`

启动中间件：

```bash
cd /root/rag_learning/gme_security_multimodal_demo
docker compose up -d
```

查看状态：

```bash
docker compose ps
```

确认 Redis 可用：

```bash
docker compose exec redis redis-cli ping
```

正常返回：

```text
PONG
```

确认 Milvus 端口可访问：

```bash
curl http://127.0.0.1:9091/healthz
```

如果返回健康状态或 HTTP 200，说明 Milvus Standalone 已启动。首次拉取镜像会比较慢。

停止中间件：

```bash
docker compose down
```

如果需要清空 Milvus / Redis 数据后重建：

```bash
docker compose down -v
docker compose up -d
```

## 4. 配置运行参数

```bash
export GME_MODEL_PATH=/models/gme-Qwen2-VL-7B-Instruct
export GME_DEVICE=cuda
export GME_IMAGE_ROOT=/data/security_multimodal_images
export GME_TRUST_REMOTE_CODE=true
export GME_NORMALIZE_EMBEDDINGS=true

export USE_MILVUS=true
export USE_REDIS=true
export MILVUS_URI=http://127.0.0.1:19530
export MILVUS_KB_COLLECTION=security_multimodal_kb
export MILVUS_MEMORY_COLLECTION=security_long_term_memory
export MILVUS_REBUILD_INDEX=true
export REDIS_URL=redis://127.0.0.1:6379/0
export REDIS_PREFIX=gme_security_rag
```

也可以参考项目根目录的 `.env.example`。如果没有 GPU，可以把 `GME_DEVICE` 临时改成 `cpu`，但 7B 多模态模型会很慢。

## 5. 准备知识库资源

文本知识放在：

```text
data/security_multimodal_docs/
```

当前程序读取该目录下的 `.md` 文件，并识别：

```md
IMAGE_REF: alarm_timeline_screenshot_01
TABLE_REF: milvus_checklist_table
BASE64_REF: b64_alarm_timeline_payload_01
```

图片放在 `GME_IMAGE_ROOT`：

```bash
mkdir -p /data/security_multimodal_images
```

`GME_IMAGE_ROOT` 用于解析知识库和查询里的图片引用。文档中的 `IMAGE_REF: alarm_timeline_screenshot_01` 会依次尝试：

- `/data/security_multimodal_images/alarm_timeline_screenshot_01`
- `/data/security_multimodal_images/alarm_timeline_screenshot_01.jpg`
- `/data/security_multimodal_images/alarm_timeline_screenshot_01.jpeg`
- `/data/security_multimodal_images/alarm_timeline_screenshot_01.png`
- `/data/security_multimodal_images/alarm_timeline_screenshot_01.webp`

接口中也可以直接传绝对路径或 `data:image/...;base64,...`。

## 6. 启动 API

```bash
cd /root/rag_learning/gme_security_multimodal_demo
uvicorn app.api:app --host 0.0.0.0 --port 8012
```

服务启动时会执行：

1. 读取 `data/security_multimodal_docs/*.md`。
2. 解析文本、图片、图文对和表格知识块。
3. 调用真实 GME 生成向量。
4. 将知识块写入 Milvus `security_multimodal_kb` 集合。
5. 初始化 Redis 短期记忆和查询缓存。

缺少模型依赖、模型目录、中间件或图片资源时，服务会直接报错，不会回退到 mock 向量。

## 7. 调用示例

健康检查：

```bash
curl http://127.0.0.1:8012/
```

文本问答：

```bash
curl --get "http://127.0.0.1:8012/ask" \
  --data-urlencode "question=Milvus 部署后检索不到结果，应该怎么排查"
```

图文问答：

```bash
curl --get "http://127.0.0.1:8012/ask" \
  --data-urlencode "question=这张告警时间线截图为什么可能是误报" \
  --data-urlencode "image_ref=/data/security_multimodal_images/alarm_timeline_screenshot_01.png"
```

纯图检索：

```bash
curl --get "http://127.0.0.1:8012/ask" \
  --data-urlencode "image_ref=/data/security_multimodal_images/gate_snapshot_01.jpg"
```

如果回答进入人工审核，返回中会有 `human_review_pending=true` 和 `thread_id`，然后调用：

```bash
curl --get "http://127.0.0.1:8012/resume" \
  --data-urlencode "thread_id=<上一步返回的 thread_id>" \
  --data-urlencode "approved=true"
```

## 8. 数据落点

- Milvus `security_multimodal_kb`：知识库 chunk 的 GME 向量和元数据。
- Milvus `security_long_term_memory`：多轮问答沉淀的长期语义记忆。
- Redis `gme_security_rag:short:*`：用户短期会话。
- Redis `gme_security_rag:cache:*`：重复问题的检索缓存。

`MILVUS_REBUILD_INDEX=true` 时，服务启动会重建知识库集合，适合开发和小规模验证。生产环境语料稳定后可以改成：

```bash
export MILVUS_REBUILD_INDEX=false
```

## 9. 常见问题

- `docker compose up -d` 失败：先确认 Docker 服务已启动，且端口 `19530`、`6379`、`9000`、`9001` 没有被占用。
- Redis 连接失败：执行 `docker compose exec redis redis-cli ping`，确认返回 `PONG`。
- Milvus 连接失败：执行 `docker compose ps`，确认 `milvus`、`etcd`、`minio` 都是 running。
- 图片找不到：确认 `GME_IMAGE_ROOT` 和 markdown 中的 `IMAGE_REF` 文件名一致。
- 首次启动很慢：7B 模型加载和知识库向量化都在首次启动时执行，语料和图片越多耗时越长。
- 内存或显存不足：先减少知识库文档和图片数量，或将 `GME_DEVICE=cpu` 做功能验证，再切回 GPU。
