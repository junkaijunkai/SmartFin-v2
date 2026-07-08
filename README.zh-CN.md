# SmartFin

> **AI 多智能体个人财务助手**

SmartFin 是一个多智能体（Multi-Agent）AI 系统，将日常消费行为与长期财务健康连接起来。基于 LangGraph 的 Supervisor 模式，协调五个专业智能体协同工作——它们共享同一个类型化的状态对象，通过工具调用（Tool Calling）进行推理，并在执行关键操作前暂停以等待用户确认。

[English Documentation](README.md)

---

## 功能特性

| 功能 | 说明 |
|---|---|
| **消费分析** | 自动对原始交易进行分类，计算各品类 30 天消费趋势 |
| **预算规划** | 基于收入和历史消费生成月度预算分配，评估执行进度，在超支前预警 |
| **目标追踪** | 从自然语言中提取财务目标（如"攒 8000 元应急基金"），计算所需月储蓄额并跟踪进度 |
| **异常检测** | 使用统计方法（IQR、频次分析）标记可疑交易，并生成自然语言解释 |
| **健康评估** | 通过负债收入比、流动储备月数、收入集中风险、持续超支等维度评估财务健康度 |
| **人机协同（HITL）** | 在关键决策点暂停执行，展示确认卡片；用户可以批准、拒绝或补充说明后再继续 |
| **持久记忆** | 跨会话记住财务数据——目标、预算、交易历史以 Markdown 文件存储，在相关对话中自动召回 |

---

## 架构设计

SmartFin 基于 **LangGraph Supervisor 模式**构建。一个编排节点（Supervisor）负责分类用户意图，并根据当前状态将控制权交给对应的专业智能体。所有数据通过共享的 `AppState` TypedDict 传递。

```
用户消息
     │
     ▼
┌────────────────┐     active_agent     ┌──────────────────────┐
│  记忆加载器     │─────────────────────▶│        Supervisor    │
│  (追踪上下文 +  │                      │   (意图分类器)       │
│   记忆召回)     │                      └──────────┬───────────┘
└────────────────┘                                 │
                                                route_to_agent()
         ┌──────────────┬─────────────┬──────────────┼──────────────┬──────────────┐
         ▼              ▼             ▼              ▼              ▼              ▼
     消费分析        预算规划       目标设定       异常检测        健康评估
         │              │             │              │              │
         └──────────────┴─────────────┴──────────────┴──────────────┘
                                              │
                                     pending_confirmation?
                                              │
                                              ▼
                                     ┌────────────────┐
                                     │  HITL 确认节点  │ ← interrupt_before
                                     │  (用户审核)     │
                                     └────────────────┘
                                              │
                                              ▼
                                     记忆保存器
                                     (持久化状态)
```

### 图执行流程

每一轮对话遵循相同的流程：

1. **记忆加载器** — 设置追踪上下文（trace ID、span ID），基于用户最新消息召回相关长期记忆（之前会话中的目标、预算等）。
2. **Supervisor** — 通过 LLM 对用户意图进行分类。当有新交易数据时，优先路由到 `expense_analysis` 进行数据预处理，并将用户的真实意图暂存为 `pending_intent`。智能体执行完毕后，Supervisor 决定继续路由下一个智能体还是结束。
3. **专业智能体** — 被路由到的智能体运行其 ReAct 循环，调用确定性工具，将结果写入共享状态。
4. **记忆保存器** — 将状态持久化到文件系统备份，如果智能体输出已确认（非暂态），则写入长期记忆。
5. **HITL 确认节点** — 如果智能体请求了用户确认，图在此处通过 `interrupt_before` 暂停。UI 展示确认卡片，用户的决策通过一次原子请求恢复执行。

### 共享状态

所有智能体通过一个 `AppState` TypedDict 通信（定义在 `app/state.py`）。每个智能体在 `app/orchestrator/state_view.py` 中声明它可以读写的字段——这是一个运行时强制的作用域，防止一个智能体污染另一个智能体的数据：

```
transactions ──▶ categorised_transactions ──▶ spending_trends
                                          ──▶ budget_allocations
                                          ──▶ anomaly_flags
                                          ──▶ health_summary
```

如果智能体尝试读取或写入其声明作用域之外的字段，会立即抛出 `KeyError` / `ValueError`。

---

## 智能体设计

每个专业智能体遵循 **ReAct（推理 + 行动）** 模式：LLM 反复推理用户请求、调用确定性工具、观察结果，最终生成回答。

### Tool Calling 机制

1. 每个智能体使用 `@tool` 装饰器定义一组工具函数，每个工具执行一个明确的业务逻辑（如计算统计指标、验证数据）。
2. 智能体通过 `.bind_tools()` 将工具绑定到 `get_llm()` 返回的 LangChain chat model。
3. `run_react_loop()`（`app/agents/react_utils.py`）管理整个循环：发送系统提示和用户消息、调用 LLM、将工具调用路由到对应函数、将结果反馈给 LLM。
4. 共享的 `final_answer` 工具标志执行完成。其结构化参数（`summary`、`needs_hitl_confirmation`、`hitl_summary`、`hitl_details`）被循环捕获并写入 `pending_confirmation`，触发 HITL 流程。

### 各智能体工具列表

| 智能体 | 工具 |
|---|---|
| **消费分析** | `categorise_transactions_tool`、`compute_spending_trends_tool` |
| **预算规划** | `extract_budget_request_tool`、`generate_allocations_tool`、`calculate_spending_tool`、`evaluate_progress_tool`、`generate_warnings_tool`、`validate_budget_tool` |
| **目标设定** | `extract_goal_tool`、`create_goal_tool`、`calculate_required_saving_tool` |
| **异常检测** | `run_statistical_detection_tool`、`generate_explanation_tool` |
| **健康评估** | `compute_health_assessment_tool` |

---

## 人机协同（HITL）

关键财务决策会在执行前暂停等待用户确认：

1. 智能体在 `final_answer` 调用中设置 `needs_hitl_confirmation=True`。图编译时指定了 `interrupt_before=["confirm"]`，因此执行会在 `confirm` 节点前暂停。
2. UI 收到一个 `__pause__` SSE 事件，其中包含 `pending_confirmation` 负载（操作类型、摘要、详情）。
3. 用户点击批准、拒绝，或输入澄清文本。
4. UI 发送 `POST /threads/{id}/runs/stream`，并在请求体中包含 `resume` 字段。图原子式恢复执行——不再需要单独的 PATCH 请求，消除了恢复与读取状态之间的竞态条件。

---

## 记忆系统

SmartFin 将财务数据以 **Markdown 文件 + YAML 前置元数据** 的形式持久化到 `.smartfin/memory/` 目录：

```
.smartfin/memory/
├── MEMORY.md                    # 索引文件
├── transactions/
│   └── 2026-05.md               # 月度交易记录
├── incomes/
│   └── 2026-05.md
├── goals/
│   └── emergency-fund.md        # 每个目标一个文件
└── budgets/
    └── 2026-05-plan.md
```

每轮对话开始时，一个轻量级 LLM（2 秒超时）扫描索引文件，选择与用户当前消息相关的文件。这些文件的内容被注入到智能体的系统提示中作为用户历史上下文。系统优雅降级——如果 LLM 调用失败或超时，则不加载任何记忆。

---

## 可观测性

每个请求产生一条结构化的追踪链路：

```
trace_id (每次请求)
  └── span_id (每个图节点)
       └── parent_span_id (父子关联)
            └── 事件: STATE_SNAPSHOT, TOOL_CALL, TOKEN_USAGE, API_REQUEST, ...
```

追踪数据以 JSONL 格式写入 `.smartfin/traces/`，按 `thread_id` 分文件。Token 使用量通过 LangChain 回调处理器捕获。错误事件按类别（验证错误、工具错误、LLM 错误、内部错误）分类，便于程序化过滤。

---

## 快速开始

### 前置条件

- Python 3.11+
- Docker（推荐）或本地 Python 环境
- [Anthropic API 密钥](https://console.anthropic.com/)

### 使用 Docker 运行

```bash
git clone https://github.com/junkaijunkai/SmartFin-v2.git
cd SmartFin-v2

# 配置环境变量
cp .env.example .env
# → 填入 ANTHROPIC_API_KEY

# 启动所有服务
docker compose up --build
```

- **Streamlit UI**: `http://localhost:8501`
- **后端 API**: `http://localhost:8000`

### 本地运行

```bash
pip install -r requirements.txt
cp .env.example .env  # → 填入 ANTHROPIC_API_KEY

# 启动后端
uvicorn app.api:app --host 0.0.0.0 --port 8000

# 另一个终端，启动 UI
streamlit run ui/app.py
```

### 运行测试

```bash
pytest tests/ -v
```

---

## 技术栈

| 层 | 技术 |
|---|---|
| **智能体编排** | [LangGraph](https://github.com/langchain-ai/langgraph)（Supervisor 模式） |
| **大语言模型** | [Anthropic Claude](https://www.anthropic.com/)（默认 Haiku，可配置 Sonnet） |
| **LLM 框架** | [LangChain](https://www.langchain.com/) |
| **后端** | FastAPI |
| **UI** | Streamlit |
| **数据校验** | Pydantic v2 |
| **可观测性** | LangSmith、结构化 JSONL 日志 |
| **基础设施** | Docker Compose、PostgreSQL（状态检查点）、Redis（缓存） |

---

## 环境变量

| 变量 | 必填 | 说明 |
|---|---|---|
| `ANTHROPIC_API_KEY` | 是 | Claude API 访问密钥 |
| `LANGSMITH_API_KEY` | 否 | LangSmith 追踪 |
| `LANGSMITH_TRACING` | 否 | 设为 `true` 启用 LangSmith |
| `LANGSMITH_PROJECT` | 否 | LangSmith 项目名（默认: `smartfin`） |
| `SMARTFIN_EVAL_PROVIDER` | 否 | Capability Eval provider 类型（默认: `openai-compatible`） |
| `SMARTFIN_EVAL_BASE_URL` | 否 | Capability Eval 使用的 OpenAI-compatible endpoint |
| `SMARTFIN_EVAL_API_KEY` | 否 | Capability Eval provider API key |
| `SMARTFIN_EVAL_MODEL` | 否 | Capability Eval 被测模型（默认: `deepseek-v4-pro`） |
| `SMARTFIN_EVAL_JUDGE_MODEL` | 否 | 语义评估 judge 模型（默认: `deepseek-v4-pro`） |
| `SMARTFIN_EVAL_SUITE` | 否 | `smoke` 或 `full`，用于选择评估数据集 |
| `SMARTFIN_MODEL` | 否 | Claude 模型 ID 或别名（默认: `claude-haiku-4-5`） |
| `SMARTFIN_ENFORCE_APPROVED_MODELS` | 否 | 设为 `true` 时，未批准的模型 ID 会回退到注册表默认值 |
| `SMARTFIN_LOG_FORMAT` | 否 | `plain` 或 `json` 日志格式 |

---

## Capability Evals

Capability Eval 位于 `tests/evals`，golden dataset 位于
`tests/evals/goldens`。当前覆盖意图识别和 5 个专业 agent 组件。
结构化 L1/L3 能力使用 deterministic assertions；自然语言解释/建议类 L2
能力使用配置好的 OpenAI-compatible judge model。

本地运行：

```bash
SMARTFIN_EVAL_SUITE=smoke python -m pytest tests/evals -m eval -v
SMARTFIN_EVAL_SUITE=full python -m pytest tests/evals -m eval -v
```

CI 会在 PR 上运行 smoke eval，在 `main` / `dev` push 上运行 full eval。
报告生成在 `reports/evals`，并作为 GitHub Actions artifact 上传；
`reports/evals/summary.md` 也会写入 Actions Summary。

## API 端点

| 方法 | 路径 | 说明 |
|---|---|---|
| `GET` | `/health` | 服务健康检查 |
| `POST` | `/analyze` | 一次性分析（非流式，返回结构化响应） |
| `POST` | `/threads/{id}/runs/stream` | SSE 流式执行，支持通过 `resume` 负载恢复 HITL |
| `GET` | `/threads/{id}/state` | 读取当前线程状态（调试用） |

---

## 许可

本项目用于学术目的。
