# QuantaAlpha-Lite

基于 LangGraph 的轻量量化因子研究实验：根据研究方向生成假设和公式，用本地 Qlib 行情计算因子及 Rank IC，再将最近六轮反馈交给模型迭代。

```text
研究方向 → 假设 → 2 个公式 → 本地计算与评估 → 反馈 → 下一轮
```

项目处于研究阶段。评估是单因子截面相关性，不是组合回测或收益承诺。反复使用同一探索区间筛选会产生过拟合；需另行固定候选、在适当的后续样本中检验。

## 准备环境

Python 3.11；以下命令以 macOS/Linux 为例。创建独立环境或复用已有环境：

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
```

在 `.env` 中设置 `OPENAI_API_KEY`；不要将它提交到 Git。API 使用 OpenAI Responses，具体模型名称及推理强度在 `configs/research.yaml` 中配置。需要相应账户和模型访问权限；生成过程会产生 API 费用。

**本仓库不包含行情数据。** 在 `configs/research.yaml` 的 `loop.provider_uri` 指向已有 Qlib 中国市场日线数据目录，其下需要 `calendars/day.txt`、`instruments/csi300.txt`、`features/`，以及配置字段对应的行情。当前评估需要历史沪深 300 成分、2008 年末起的预热数据和完整 2020—2023 交易日历。相对路径以项目目录为基准。具体数据获取方式和授权需由使用者自行确认。

## 运行

```bash
# 只预览提示词：不访问 API、不加载行情
python -m lite --loop --dry-run

# 启动新实验
python -m lite --loop --env-file .env

# 断点续跑：替换为实际运行目录
python -m lite --resume runs/<run_id> --env-file .env
```

配置 `research.direction` 修改方向；`loop.max_rounds` 为 1—30，每轮固定 2 个候选；每轮通常两次模型调用，SDK 自动重试关闭。终端输出表达式和年度 Rank IC。单轮只生成提案可去掉 `--loop`；`--evaluation <path>` 可作为探索反馈。

续跑使用该运行保存在 `config.json` 中的配置和历史，不读取新运行的方向。完成的轮次不重新计算，缓存的响应仅在提示词完全一致时复用；不匹配时停止。修改已有运行上限需更改该运行自己的配置。请勿同时对同一个目录运行两个进程。

## 研究口径

循环评估目前固定：历史沪深 300，信号 2020—2023 年、未来 21 交易日收益 `close[T+22]/close[T+1]-1`。为避免标签进入 2024 年，2023 年最后一个信号日为 11 月 29 日。每天计算横截面 IC 和 Rank IC，再按日期平均；仅用过去和当日可用数据。

截面排名使用当日有效成分股、在未来标签过滤之前计算。时间序列窗口需要完整数据，成分资格缺失会打断窗口。没有自动进行 ST 过滤、行业或市值中性化、交易成本或涨跌停模拟。21 日标签相互重叠，逐日 IC 不是独立样本；不会自动做显著性检验。

公式使用受限 AST 解释器，不执行模型生成的 Python。当前允许 `ABS`、`SIGN`、`LOG`、`SQRT`、`POW`、`MIN`、`MAX`、`RANK`、`MEAN`、`STD`、`ZSCORE`、`SCALE`、`DELAY`、`TS_SUM`、`TS_MEAN`、`TS_STD`、`TS_MIN`、`TS_MAX`、`EMA` 和基础运算。详细语义和限制见 `lite/expressions.py` 中的 `GUIDE`；`EMA` 为 `adjust=False`，完整连续窗口后输出。非法公式会记录失败，模型格式错误等异常会停止当前运行。

完整历史保存在本地；每次调用只包含最近六条历史的详细反馈。`runs/` 是本地运行产物，`evaluations/` 是独立分析结果，均不随仓库上传；每轮包含实际提示词、模型响应、假设、因子、`evaluation.json`、每日指标和因子值。

## 离线测试

```bash
python -m unittest discover -s tests -v
```

测试使用模拟模型与合成数据，不请求 API、不运行真实研究循环。若同级未克隆原项目，上游 prompt 一致性测试会跳过。仓库代码在现有 macOS/Python 3.11 环境通过测试；全新环境安装及其他平台尚未验证。

## 上游来源

本项目参考 [QuantaAlpha](https://github.com/QuantaAlpha/QuantaAlpha)。`prompts/original.yaml` 的七组提示词文本复制自上游；来源路径和 SHA256 见 `prompts/source.json`。上游仓库截至整理时未声明许可证，本项目未为上游内容另行声明开源许可。
