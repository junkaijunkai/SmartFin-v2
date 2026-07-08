# GitHub Workflows

这个目录存放 SmartFin 的 GitHub Actions 工作流，当前包含两个核心流程：

- `ci-llmsecops.yml`
- `cd-deploy.yml`

## CI: `ci-llmsecops.yml`

### 作用

CI 用来在代码变更时自动做质量与安全检查，避免有问题的代码进入主分支。

### 触发时机

CI 会在以下情况触发：

- `push` 到 `main`
- `push` 到 `master`
- `push` 到 `dev`
- 任意 `pull_request`

### 当前执行内容

CI 主要执行以下步骤：

- 检出代码
- 安装 Python 3.11 与依赖
- 运行 `ruff` 静态检查
- 运行 `tests/unit`
- 运行 `tests/integration`
- 运行 `tests/security`
- 运行 Capability Eval：PR 使用 smoke dataset，`main` / `dev` push 使用 full dataset
- 运行 `scripts/llmsecops_ci.py`
- 运行 `bandit` 安全扫描
- 运行 `pip-audit` 依赖漏洞扫描
- 构建后端 Docker 镜像
- 构建前端 Docker 镜像

### 目标

CI 的目标是尽量提前发现：

- 代码风格和静态问题
- 单元测试 / 集成测试失败
- Agent capability 回归（意图识别、结构化抽取、分类、解释/建议质量）
- 安全策略回归
- 依赖漏洞
- Docker 构建问题

## CD: `cd-deploy.yml`

### 作用

CD 用来在需要时手动重部署当前分支代码到目标服务器。

当前部署目标：

- 服务器目录：`/opt/smartfin`
- 部署方式：通过 SSH / SCP 上传源码，然后在服务器上执行 `docker compose up -d --build`

### 触发时机

CD 当前只支持一种触发方式：

- 手动触发：在 GitHub Actions 页面通过 `workflow_dispatch` 手动执行

这意味着：

- `push` 到 `main` / `master` / `dev`，只会运行 CI 检查和构建，不会自动部署
- `pull_request` 即使触发了 CI，也不会自动部署
- `cd-deploy.yml` 主要用于手动重试或手动重部署

### 当前执行内容

CD 主要执行以下步骤：

- 检出要部署的代码版本
- 通过 SSH 在服务器上准备部署目录
- 清理上一次上传的源码文件
- 通过 SCP 上传运行所需源码和配置文件
- 在服务器上生成 `.env`
- 执行 `docker compose up -d --build --remove-orphans`

### 并发控制

CD 配置了 `concurrency`，同一个分支新的部署会取消旧的进行中部署，避免同分支重复部署互相覆盖。

## 需要的 GitHub Secrets

### 部署连接

- `SERVER_HOST`
- `SERVER_USER`
- `SERVER_PASSWORD`

### 运行时配置

必填：

- `ANTHROPIC_API_KEY`

可选：

- `LANGSMITH_API_KEY`
- `LANGSMITH_TRACING`
- `LANGSMITH_PROJECT`
- `SMARTFIN_MODEL`
- `SMARTFIN_ENFORCE_APPROVED_MODELS`
- `LOG_LEVEL`
- `SMARTFIN_LOG_FORMAT`

### Capability Eval 配置

CI 中的 Capability Eval 绕过 LiteLLM Gateway，直接访问 OpenAI-compatible
模型服务。需要在 GitHub Secrets 中配置：

- `SMARTFIN_EVAL_BASE_URL`
- `SMARTFIN_EVAL_API_KEY`

可选：

- `SMARTFIN_EVAL_PROVIDER`（默认: `openai-compatible`）
- `SMARTFIN_EVAL_MODEL`（默认: `deepseek-v4-pro`）
- `SMARTFIN_EVAL_JUDGE_MODEL`（默认: `deepseek-v4-pro`）

PR 会设置 `SMARTFIN_EVAL_SUITE=smoke`；`main` / `dev` push 会设置
`SMARTFIN_EVAL_SUITE=full`。Eval 报告会生成在 `reports/evals`，上传为
GitHub Actions artifact，并写入 Actions Summary。

## 维护建议

- 修改 CI 步骤时，同步更新本 README
- 修改 CD 触发条件或部署目录时，同步更新本 README
- 如果后续区分测试环境 / 生产环境，建议拆分独立 CD workflow
