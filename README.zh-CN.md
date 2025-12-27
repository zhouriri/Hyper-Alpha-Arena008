# <img width="40" height="40" alt="logo_app" src="https://github.com/user-attachments/assets/911ba846-a08b-4e3e-b119-ec1e78347288" style="vertical-align: middle;" /> Hyper Alpha Arena

[English](./README.md) | **简体中文**

> 开箱即用的开源 AI 交易平台。不会写代码？没关系——用自然语言就能让 AI 帮你自动交易加密货币。支持 Hyperliquid DEX（测试网模拟 + 主网实盘），Binance、Aster DEX 即将接入。

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![GitHub stars](https://img.shields.io/github/stars/HammerGPT/Hyper-Alpha-Arena)](https://github.com/HammerGPT/Hyper-Alpha-Arena/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/HammerGPT/Hyper-Alpha-Arena)](https://github.com/HammerGPT/Hyper-Alpha-Arena/network)
[![Community](https://img.shields.io/badge/Telegram-Community-blue?logo=telegram)](https://t.me/+RqxjT7Gttm9hOGEx)
[![English](https://img.shields.io/badge/Lang-English-blue)](https://www.akooi.com/docs/)
[![中文](https://img.shields.io/badge/语言-中文-red)](https://www.akooi.com/docs/zh/)

## 🔥 开始交易，最高省 30% 手续费

### 🚀 **Hyperliquid** — 去中心化合约交易所
- **无需 KYC** | **手续费低** | **性能强悍**
- 与本项目深度集成，开箱即用
- [**立即开户 →**](https://app.hyperliquid.xyz/join/HYPERSVIP)

### 💰 **Binance** — 全球最大交易所
- **手续费打 7 折** | **深度好** | **工具全**
- [**注册领 30% 返佣 →**](https://accounts.maxweb.red/register?ref=HYPERVIP)

### ⚡ **Aster DEX** — 兼容 Binance 的 DEX
- **手续费更低** | **多链支持** | **API 钱包更安全**
- [**立即注册 →**](https://www.asterdex.com/zh-CN/referral/2b5924)

---

## 这是什么

Hyper Alpha Arena 是一个 AI 交易平台——让 GPT、Claude、Deepseek 这些大模型帮你盯盘、分析、下单，全自动。灵感来自 [nof1 Alpha Arena](https://nof1.ai)。

**官网：** https://www.akooi.com/

## 适合谁用

| 你是谁 | 能得到什么 |
|--------|-----------|
| **普通交易者** | 不会写代码也能用，AI 助手陪你聊天就能配好策略 |
| **量化玩家** | 先在测试网跑通策略，再上真金白银 |
| **Hyperliquid 用户** | 测试网免费练手，主网 1-50 倍杠杆实盘 |
| **AI 发烧友** | 让 GPT、Claude、Deepseek 同台 PK，看谁更能赚 |

**两种模式：**
- **测试网（模拟盘）**：零风险练手，真实行情、真实订单簿、免费测试金
- **主网（实盘）**：真金白银，1-50 倍杠杆，盈亏自负

## 核心功能

### 交易能力
- **多模型支持**：GPT、Claude、Deepseek……兼容所有 OpenAI API 格式的模型
- **多钱包架构**：每个 AI Trader 可以分别配置测试网和主网钱包
- **全局环境切换**：一键切换所有 Trader 的交易环境
- **AI 提示词生成器**：
  - 用自然语言描述你的交易想法
  - AI 自动生成结构化的策略提示词
  - 支持对话历史管理
  - 一键应用到 Trader 配置
- **提示词模板管理**：
  - 可视化编辑器，自定义交易提示词
  - 支持账户级别的模板绑定
  - 内置默认、专业版、Hyperliquid 专用模板
- **技术指标**：内置 11 个常用指标
  - 趋势类：SMA、EMA、MACD
  - 动量类：RSI、随机指标
  - 波动类：布林带、ATR
  - 成交量：OBV、VWAP
  - 支撑阻力：枢轴点、斐波那契
- **信号触发交易**：
  - 自定义触发条件：OI 变化、资金费率异动、价格突破……
  - AI 信号生成器：用自然语言描述，自动转成信号配置
  - 可与定时触发组合使用
- **多 Trader 管理**：创建多个 AI Trader，各自独立配置
- **AI 归因分析**：看清哪笔赚、哪笔亏
  - 按币种、策略、触发类型拆解收益
  - 胜率、盈亏比、PnL 追踪
  - AI 辅助诊断，找出策略短板

### Hyperliquid 专属功能
- **永续合约交易**：
  - 市价单、限价单，1-50 倍杠杆
  - 多空双向，自动计算强平价
  - 全仓模式，实时监控保证金
- **环境隔离**：
  - 测试网、主网严格分开
  - 钱包、缓存、API 调用完全隔离
- **风控机制**：
  - 最大杠杆限制（可按账户配置）
  - 保证金使用率超 80% 自动暂停
  - 强平价格预警
- **AI 驱动交易**：
  - 杠杆感知提示词，内置风控教育
  - AI 根据市场信心自动选择杠杆倍数
  - 与现有 AI 决策引擎无缝集成

## 界面预览

### 仪表盘总览
![仪表盘总览](screenshots/dashboard-overview.png)

### AI 提示词生成器
![AI 提示词生成器](screenshots/ai-prompt-generator.png)

### 技术分析
![技术分析](screenshots/ai-technical-analysis.png)

### Trader 配置
![Trader 配置](screenshots/trader-configuration.png)

## 快速开始

### 环境要求

- **Docker Desktop**（[下载地址](https://www.docker.com/products/docker-desktop)）
  - Windows：Docker Desktop for Windows
  - macOS：Docker Desktop for Mac
  - Linux：Docker Engine（[安装指南](https://docs.docker.com/engine/install/)）

### 安装步骤

```bash
# 克隆仓库
git clone https://github.com/HammerGPT/Hyper-Alpha-Arena.git
cd Hyper-Alpha-Arena

# 启动应用（根据你的 Docker 版本选择命令）
docker compose up -d --build        # 新版 Docker Desktop（推荐）
# 或
docker-compose up -d --build       # 旧版 Docker 或独立安装的 docker-compose
```

启动完成后访问 **http://localhost:8802**

### 常用命令

```bash
# 查看日志
docker compose logs -f        # 或 docker-compose logs -f

# 停止应用
docker compose down          # 或 docker-compose down

# 重启应用
docker compose restart       # 或 docker-compose restart

# 更新到最新版本
git pull origin main
docker compose up -d --build # 或 docker-compose up -d --build
```

**注意事项**：
- 所有数据（数据库、配置、交易记录）都保存在 Docker 卷中
- 停止/重启容器不会丢失数据
- 只有 `docker-compose down -v` 会删除数据（除非想重置，否则别加 `-v`）

## 首次配置

详细配置指南请参考官方文档，包括：
- Hyperliquid 钱包配置（测试网 & 主网）
- AI Trader 创建与 LLM API 设置
- 交易环境与杠杆设置
- 信号触发交易配置

**📖 完整指南：[快速开始](https://www.akooi.com/docs/zh/guide/getting-started.html)**

## 支持的模型

Hyper Alpha Arena 支持所有兼容 OpenAI API 的大语言模型。**推荐使用 Deepseek**，性价比高，交易场景表现出色。

支持的模型包括：
- **Deepseek**（推荐）：交易决策性价比之王
- **OpenAI**：GPT-5 系列、o1 系列、GPT-4o、GPT-4
- **Anthropic**：Claude（通过兼容端点）
- **自定义 API**：任何 OpenAI 兼容的端点

平台会自动处理不同模型的配置差异。

## 常见问题

**问题**：端口 8802 被占用
**解决**：
```bash
docker-compose down
docker-compose up -d --build
```

**问题**：无法连接 Docker 守护进程
**解决**：确保 Docker Desktop 正在运行

**问题**：数据库连接错误
**解决**：等待 PostgreSQL 容器启动完成（用 `docker-compose ps` 检查状态）

**问题**：想要重置所有数据
**解决**：
```bash
docker-compose down -v  # 这会删除所有数据！
docker-compose up -d --build
```

## 参与贡献

欢迎社区贡献！你可以：

- 报告 Bug 和问题
- 提出新功能建议
- 提交 Pull Request
- 完善文档
- 在不同平台测试

请 Star 和 Fork 本仓库，关注开发进展。

## 相关资源

### Hyperliquid
- 官方文档：https://hyperliquid.gitbook.io/
- Python SDK：https://github.com/hyperliquid-dex/hyperliquid-python-sdk
- 测试网：https://api.hyperliquid-testnet.xyz

### 原始项目
- Open Alpha Arena：https://github.com/etrobot/open-alpha-arena

## 社区与支持

**🌐 官网**：[https://www.akooi.com/](https://www.akooi.com/)

**🐦 Twitter/X**：[@GptHammer3309](https://x.com/GptHammer3309)
- Hyper Alpha Arena 最新动态
- AI 交易见解与策略讨论
- 技术支持与答疑

**💬 Telegram 群**：[点击加入](https://t.me/+RqxjT7Gttm9hOGEx)
- 反馈 Bug（尽量附日志、截图、复现步骤）
- 讨论策略或产品体验
- PR / Issue 想要我关注可在群里提醒

注意：Telegram 主要用于快速沟通，正式记录请继续使用 GitHub Issues / Pull Requests；谨记不要分享密钥等敏感信息。

## 许可证

本项目采用 Apache License 2.0 许可证。详见 [LICENSE](LICENSE) 文件。

## 致谢

- **etrobot** - open-alpha-arena 原始项目
- **nof1.ai** - Alpha Arena 灵感来源
- **Hyperliquid** - 去中心化永续合约交易平台
- **OpenAI、Anthropic、Deepseek** - LLM 提供商

---

Star 本仓库，关注开发进展。