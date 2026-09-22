# examples/ — 接入示例

给**不用 Claude Code**、或需要无人值守跑的人。完整部署 SOP 见仓库根的 [`AGENTS.md`](../AGENTS.md)。

| 文件 | 干什么 | 平台 |
|---|---|---|
| [`bridge_any_llm.py`](bridge_any_llm.py) | 把任意 OpenAI 兼容模型(GPT/DeepSeek/Gemini/GLM/Kimi/通义/本地…)接成 AI 侧 | 任意 |
| [`api_loop.py`](api_loop.py) | 服务器常驻 API 身体：任意 OpenAI-compatible 模型、看图、MCP 多步工具、附件收发、人格热加载；配合 PWA 的 Desktop/API 开关、设置页、多窗口和流式输出 | Linux/VPS |
| [`mcp_demo_server.py`](mcp_demo_server.py) | 三个工具的最小 MCP server（stdio / `--http PORT`），用来验证 api_loop 的工具链路 | 任意 |
| [`requirements.txt`](requirements.txt) | `api_loop.py` 的依赖（fastapi / uvicorn / httpx / mcp） | — |
| [`persona.example.md`](persona.example.md) · [`api_loop.config.example.json`](api_loop.config.example.json) | 人格文件与运行态配置的样例 | — |
| [`companion-api-loop.service`](companion-api-loop.service) | `api_loop.py` 的 systemd 模板 | Linux/VPS |
| [`.env.example`](.env.example) | `bridge_any_llm.py` / `api_loop.py` 共用配置模板 | — |
| [`confirm_dev_channel_win.py`](confirm_dev_channel_win.py) | Windows 上自动确认 Claude Code 的 DevChannelsDialog 弹框 | Windows |

---

## 用任意 LLM 当大脑(bridge_any_llm.py)

它替代 `channel/` 插件,不依赖 Claude Code。原理是个三步薄循环:SSE 收
`/channel/in` → 拉历史拼 messages + 调你的模型 → POST `/channel/out`。零第三方依赖。

```bash
cd examples
cp .env.example .env
#  编辑 .env:填 RELAY_URL、RELAY_SECRET(和后端一致),以及你的模型三件套
#  LLM_API_BASE / LLM_API_KEY / LLM_MODEL(各家取值见 .env.example 里的注释表)
python3 bridge_any_llm.py
```

跑起来后,在手机 PWA 发一条 → 终端打印 `[in] #.. ` → 模型生成 → 手机收到回复。

- **换模型**只改 `.env` 的三件套,代码不动。Gemini 用它的 OpenAI 兼容端点即可。
- **兜底链**:填 `LLM_*_2` / `LLM_*_3`,主模型 401/403/429/5xx 时自动顺次切。
- **失忆?** 调大 `HISTORY_N`(默认喂最近 12 条)。
- **看图**:默认把附件降级成文字提示;要真看图,在 `handle_human_message` 里下载
  `/uploads/{name}?token=` 再按多模态格式喂(代码里有注释标位置)。

---

## 服务器 API 身体(api_loop.py)

它不是长连消费 `/channel/in`，而是一个本机 HTTP 服务：relay 的 `/app/brain` 切到
`loop` 后，`/app/send` 会把新消息（含附件）POST 到 `/loop/ingest`，它跑完一整轮再把回复
POST 回 `/channel/out`。一轮里发生的事，按顺序：

1. **人格** — system prompt 从 `persona_file` 读；文件 mtime 变了就重读。设置页「API · 人格设定」保存 = 写文件 = 下一条生效。
2. **上文** — 直接查 relay.db，取同一 `api_session` 最近 `history_n` 条（设置页可调，0 = 无记忆）。最近 `history_images` 条带图的人类消息重新发像素，其余图只留 `[图片: 名字]`。
3. **附件** — 图片 → base64 `image_url` 交给多模态模型（模型不吃图返回 400 时自动降级成文字提示重试一次）；≤64KB 的文本类文件内联进消息；其它文件写一行说明并在 `loop_cache/in/` 留副本给工具读。
4. **工具** — 启动时连上配置的 MCP server（stdio 或 streamable-http），`tools/list` 转成 OpenAI function calling；模型出 `tool_calls` 就执行、把结果喂回去，循环到它给出正文或达到 `max_tool_steps`。每一轮在 PWA 里显示为一张「act」小卡（点开看 tool / cmd / result）。内建一个 `attach_file` 工具：模型把服务器上的文件作为附件随回复发出；MCP 工具返回的图片也会自动变成附件。
5. **回复** — `reply_delta` 流式草稿，结束落一条正式 `reply`（带 `attachments` 和 `api` 元数据：用的模型、兜底自哪条、工具步数、usage、错误）。模型链全挂时错误原文直接进气泡，不会卡在「正在输入」。

```bash
cd examples
pip install -r requirements.txt
cp .env.example .env
# 填 RELAY_URL / RELAY_SECRET / RELAY_DB / RELAY_UPLOAD_DIR；模型和人格可以不填,去设置页
python3 api_loop.py
```

把 relay 切到它（或在 PWA 设置页「联系 Claude」点 **API**）：

```bash
curl -s -X POST http://127.0.0.1:3011/app/brain \
  -H "Authorization: Bearer $RELAY_SECRET" \
  -H "Content-Type: application/json" \
  -d '{"target":"loop"}'
```

### 设置页(PWA → 头像 → 个人信息)

relay 把 `/app/loop/{path}` 原样透传到 loop 的 `/loop/{path}`，所以下面每一项都不用登服务器：

| 卡片 | 做什么 | 背后的端点 |
|---|---|---|
| API · 模型 | 模型链每行 url / model / key；**保存并测试**会真发一句 `pong` 回来看延迟和错误原文。key 回显永远打码，留空不改 | `GET/POST /loop/config` · `POST /loop/test` |
| API · 上文与工具 | `history_n` 上文条数 · `history_images` 重发几张图 · `max_tool_steps` 工具步数 · 看图开关 | `POST /loop/config` |
| API · 人格设定 | 编辑框 = persona 文件；保存即热加载 | `GET/POST /loop/persona` |
| API · MCP 工具 | 增删改 MCP server、看在线状态和工具数、重连 | `GET/POST /loop/mcp` · `DELETE /loop/mcp/{name}` · `POST /loop/mcp/{name}/reconnect` · `GET /loop/tools` |

### 先用自带的 demo 验证工具链

```bash
# 设置页 → API · MCP 工具:名字 demo · 类型 stdio · 命令 python3 · 参数 mcp_demo_server.py → 添加
# 然后在聊天里发:「把 3.5 和 4.25 加起来,写成一张叫 sum 的便签发给我」
# 应该看到三张 act 小卡(add → write_note → attach_file),回复里带一个 sum.txt 附件。
```

同一个文件也能当 streamable-http server 跑：`python3 mcp_demo_server.py --http 8765`，设置页类型选 http，URL 填 `http://127.0.0.1:8765/mcp`。

### 配置落在哪

- 运行态全部在 `api_loop.config.json`（默认在 examples/，`LOOP_CONFIG` 可改）：模型链、`history_n`、`persona_file`、`mcp_servers`、session 列表。**里面有 key，已 git-ignore，权限 600。** 样例见 `api_loop.config.example.json`。
- `.env` 只放 relay 相关和几个启动参数；`LLM_*` / `PERSONA*` 仍然认，是设置页没填时的兜底。
- `attach_file` 永远拒绝名字像密钥的文件（`.env`、`*.pem`、`*.key`、`relay.db`、`api_loop.config.json`、含 secret/token/credential 的）；想再收紧就在 config 里填 `attach_roots` 白名单目录。

### 排错

| 症状 | 看哪 |
|---|---|
| 设置页显示「API loop 没连上」 | `api_loop.py` 没跑，或 relay 的 `RELAY_LOOP_INGEST_URL` 端口和 `LOOP_PORT` 不一致。`curl 127.0.0.1:3020/healthz` |
| 保存并测试 ✗ HTTP 401 | key 错；错误原文就是供应商返回的 |
| 保存并测试 ✗ ConnectError | 服务器出不了网或需要代理。模型调用遵守 `HTTPS_PROXY` / `NO_PROXY` 环境变量（对 relay 的本机调用不走代理） |
| 模型看不到图 | 该模型不支持多模态（loop 已自动降级成文字提示）；换一个支持图片的模型 |
| MCP 状态 error | 卡片上有错误原文；stdio 的 command 要在服务器上能直接执行，相对路径相对于 examples/ |
| 工具调了但没回正文 | 部分中转把最终回答放在 `reasoning_content` 里，loop 已兼容；仍空的话把 `max_tool_steps` 调小看是不是一直在循环 |

---

## Claude Code 的确认框自动过(无人值守)

只有走 Claude Code 这条路才有这个框。**Linux/macOS 用 tmux 最干净:**

```bash
tmux new-session -d -s cc 'claude --dangerously-load-development-channels server:companion'
sleep 3 && tmux send-keys -t cc Enter        # 替你确认 DevChannelsDialog
```

**Windows(无 tmux)** 用 [`confirm_dev_channel_win.py`](confirm_dev_channel_win.py):

```bash
python confirm_dev_channel_win.py -- claude --dangerously-load-development-channels server:companion
```

细节(为什么躲不掉、覆盖范围)见 [`AGENTS.md` §4](../AGENTS.md)。

---

## ⚠️ 单身体原则

relay 是单用户单通道。**同一时刻只跑一个 AI 侧** —— 别同时开着 Claude Code channel
和 `bridge_any_llm.py`。`api_loop.py` 由 relay 的 Desktop/API 开关控流，切到 `loop`
时 Desktop channel 不会收到新消息；切回 `desktop` 时 API loop 仍可运行但不会接新入站。
