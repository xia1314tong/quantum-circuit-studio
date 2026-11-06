"""创新点 2 的实验：LLM 生成量子电路的"校验—反馈—修复"闭环。

对照设计（三个 arm，--arms 选）
------------------------------
* ``oneshot``：只问一次，直接看结果（等价于目前产品里 AI 助手的行为）；
* ``resample``：失败后用**全新会话**重问同一题面、不给任何错误信息，最多 k 次——
  这是"多试几次取成功"的公平基线，用来把"反馈的信息量"与"采样的次数"分开；
* ``feedback``：把三层校验的具体错误回传，让模型改，最多 k 轮。

只有把 ``resample`` 与 ``feedback`` 放在一起比，才能说明闭环的收益不是"多调了几次 API"。
配对统计见 ``compare_arms``（McNemar 精确检验 + bootstrap 置信区间）。

三层校验（都写在这个文件里，可独立测试）
---------------------------------------
1. **语法层**：能否解析成 JSON、字段与类型是否合法（门名、比特下标、CNOT 的
   control/target 是否同线、同一格是否重复放门）；
2. **语义层**：比特数是否符合要求、是否命中题目给的布局约束；
3. **数值层**：用本项目引擎算出末态，与该任务的期望态比较 ``|<psi|phi>|^2``。

用法::

    python -m experiments.llm_loop --reps 3 --rounds 3
    python -m experiments.llm_loop --mock        # 不调用 API，用离线假数据跑通流程

API key 从 ``DEEPSEEK_API_KEY`` 环境变量或项目根目录的 ``deepseek_config.json``
读取（该文件不入库）。

换厂商不用改代码，改三个参数即可：

    python -m experiments.llm_loop --base-url https://dashscope.aliyuncs.com/compatible-mode/v1 \
        --api-key-env DASHSCOPE_API_KEY --model qwen-plus --tag qwen

其他 OpenAI 兼容服务同理（OpenAI 官方、Ollama、vLLM 等）。
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:  # 允许以脚本方式运行
    if __package__ in (None, ""):
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from experiments import llm_tasks
    else:
        from . import llm_tasks
except Exception:  # pragma: no cover
    import llm_tasks  # type: ignore

from quantum_circuit_studio.engine import simulate_circuit

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(ROOT, "experiments", "results")

GATE_NAMES = ("H", "X", "Y", "Z", "RX", "RY", "RZ", "CNOT")
ROTATIONS = ("RX", "RY", "RZ")
MAX_COLUMN = 32

# 单次调用失败（网络抖动、限流、余额）时的重试策略：宁可慢，也不要把一次
# 偶发错误写成"模型答错了"。
API_ATTEMPTS = 3
API_RETRY_SECONDS = 20.0
# 断网时单个配对最多等这么久（期间按退避不断重试），超时才记 api_error
NETWORK_WAIT_SECONDS = 900.0
# 全部跑完后针对 api_error 的自动补跑轮数（网络恢复后可以把它们洗掉）
REPAIR_ROUNDS = 3
REPAIR_WAIT_SECONDS = 60.0

# 单次生成的最大输出预算。推理型模型（deepseek-v4-pro 等）会把大量 token 花在
# reasoning 上，预算给小了正文会被截断成空字符串；换厂商时可用 --max-tokens 调。
MAX_OUTPUT_TOKENS = 16000

SYSTEM_PROMPT = """你是量子电路生成器。你只能输出一个 JSON 对象，不要输出解释、不要用代码块包裹。

JSON 结构：
{"num_wires": <整数>, "gates": [<门>, ...]}

门的写法（二选一）：
- 单比特门：{"name": "H"|"X"|"Y"|"Z"|"RX"|"RY"|"RZ", "wire": <整数>, "col": <整数>, "param": <浮点数，仅旋转门需要>}
- 双比特门：{"name": "CNOT", "control": <整数>, "target": <整数>, "col": <整数>}

约定：比特下标从 0 开始；col 是列号，同一列可以放多个门，但不能有两个门占用同一个比特；
CNOT 的 control 与 target 不能相同。旋转门的角度参数 param 用弧度。"""


# --------------------------------------------------------------------------
# 三层校验
# --------------------------------------------------------------------------
def parse_json(text: str) -> Tuple[Optional[dict], List[str]]:
    """从模型输出里尽量抠出 JSON（容忍 ```json 包裹与前后废话）。"""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z]*\n?", "", cleaned)
        cleaned = re.sub(r"\n?```$", "", cleaned).strip()
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return None, ["输出里找不到 JSON 对象"]
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError as error:
            return None, [f"JSON 解析失败：{error.msg}（位置 {error.pos}）"]
    if not isinstance(payload, dict):
        return None, ["顶层必须是 JSON 对象，实际是 " + type(payload).__name__]
    return payload, []


def validate_syntax(payload: dict) -> Tuple[List[dict], List[str]]:
    """第 1 层：字段、类型、格子占用。返回 (规范化的门列表, 错误列表)。"""
    errors: List[str] = []
    gates = payload.get("gates")
    if not isinstance(gates, list) or not gates:
        return [], ["gates 必须是非空数组"]
    if not isinstance(payload.get("num_wires"), int):
        errors.append("num_wires 必须是整数")

    normalised: List[dict] = []
    occupied: Dict[Tuple[int, int], int] = {}
    for index, raw in enumerate(gates):
        where = f"第 {index} 个门"
        if not isinstance(raw, dict):
            errors.append(f"{where} 不是对象")
            continue
        name = raw.get("name")
        if name not in GATE_NAMES:
            errors.append(f"{where} 的门名 {name!r} 不合法，只支持 {', '.join(GATE_NAMES)}")
            continue
        col = raw.get("col")
        if not isinstance(col, int) or col < 0 or col > MAX_COLUMN:
            errors.append(f"{where} 的 col 必须是 0..{MAX_COLUMN} 的整数")
            continue

        if name == "CNOT":
            control, target = raw.get("control"), raw.get("target")
            if not isinstance(control, int) or not isinstance(target, int):
                errors.append(f"{where}（CNOT）的 control/target 必须是整数")
                continue
            if control == target:
                errors.append(f"{where}（CNOT）的 control 与 target 相同")
                continue
            cells = [(control, col), (target, col)]
            entry = {"name": "CNOT", "control": control, "target": target, "col": col}
        else:
            wire = raw.get("wire")
            if not isinstance(wire, int):
                errors.append(f"{where} 的 wire 必须是整数")
                continue
            param = raw.get("param", 0.0)
            if name in ROTATIONS:
                if not isinstance(param, (int, float)):
                    errors.append(f"{where}（{name}）缺少数值参数 param")
                    continue
                entry = {"name": name, "wire": wire, "col": col, "param": float(param)}
            else:
                entry = {"name": name, "wire": wire, "col": col, "param": 0.0}
            cells = [(wire, col)]

        clash = [cell for cell in cells if cell in occupied]
        if clash:
            errors.append(
                f"{where} 与已有门在同一格的同一比特上冲突：{clash[0][0]} 号比特第 {col} 列"
            )
            continue
        for cell in cells:
            occupied[cell] = index
        normalised.append(entry)

    return normalised, errors


def validate_semantics(
    payload: dict, gates: Sequence[dict], task: "llm_tasks.Task"
) -> List[str]:
    """第 2 层：比特数范围与题目自带的布局约束。"""
    errors: List[str] = []
    num_wires = payload.get("num_wires")
    if isinstance(num_wires, int):
        if num_wires != task.num_wires:
            errors.append(f"比特数应为 {task.num_wires}，实际给出 {num_wires}")
        for gate in gates:
            for key in ("wire", "control", "target"):
                value = gate.get(key)
                if isinstance(value, int) and not (0 <= value < num_wires):
                    errors.append(f"门 {gate['name']} 引用了越界比特 {value}（范围 0..{num_wires - 1}）")
    if task.layout is not None:
        errors.extend(task.layout(list(gates)))
    return errors


def validate_numerics(gates: Sequence[dict], task: "llm_tasks.Task") -> List[str]:
    """第 3 层：末态与期望态的重叠。"""
    try:
        result = simulate_circuit(task.num_wires, list(gates), shots=1000)
    except Exception as error:  # 引擎拒绝执行（例如门参数非法）
        return [f"仿真失败：{type(error).__name__}: {error}"]
    state = np.asarray(result.state, dtype=complex)
    expected = task.expected_state()
    overlap = float(abs(np.vdot(expected, state)) ** 2)
    if overlap < 0.99:
        return [f"末态与期望态的重叠只有 {overlap:.4f}（要求 ≥ 0.99）"]
    return []


def validate(payload: dict, task: "llm_tasks.Task") -> Dict[str, object]:
    gates, syntax_errors = validate_syntax(payload)
    semantic_errors = validate_semantics(payload, gates, task) if not syntax_errors else []
    numeric_errors = (
        validate_numerics(gates, task) if not syntax_errors and not semantic_errors else []
    )
    return {
        "gates": gates,
        "syntax": syntax_errors,
        "semantic": semantic_errors,
        "numeric": numeric_errors,
        "passed": not (syntax_errors or semantic_errors or numeric_errors),
    }


def failure_layer(report: Dict[str, object]) -> str:
    if report["passed"]:
        return "pass"
    for layer in ("syntax", "semantic", "numeric"):
        if report[layer]:
            return layer
    return "unknown"


def feedback_text(report: Dict[str, object]) -> str:
    if report.get("layer_override") == "truncated":
        # 推理型模型的失败形态：思考吃掉了全部补全预算，正文为空。
        # 这种"修复"要靠显式的预算提示，而不是指出电路哪里错。
        return (
            "你上一次的回答把全部输出预算都花在推理上，正文是空的，所以没有被采用。"
            "这一次请直接给出 JSON 对象，不要写任何推理过程；"
            "即使不能精确制备，也要给出一个最接近的、语法合法的电路。"
        )
    lines = ["你的上一次输出没有通过校验，请修正后重新输出完整 JSON。"]
    labels = {
        "syntax": "第 1 层（语法/结构）",
        "semantic": "第 2 层（语义/约束）",
        "numeric": "第 3 层（数值/结果）",
    }
    for layer, label in labels.items():
        for error in report[layer]:
            lines.append(f"- {label}：{error}")
    lines.append("只输出修正后的 JSON 对象。")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# API 调用
# --------------------------------------------------------------------------
def load_api_key(env_names: Sequence[str] = ("DEEPSEEK_API_KEY",)) -> Optional[str]:
    """按顺序找 key：先环境变量，再项目根目录的 llm_keys.json / deepseek_config.json。

    ``env_names`` 让同一个实验脚本能跑任意 OpenAI 兼容服务（Qwen、OpenAI、Ollama…），
    只要调用方通过 ``--api-key-env`` 指定变量名即可，不必改代码。
    """
    for name in env_names:
        key = os.environ.get(name)
        if key:
            return key.strip()
    for directory in (ROOT, os.path.dirname(ROOT)):
        keys_path = os.path.join(directory, "llm_keys.json")
        if os.path.exists(keys_path):
            try:
                with open(keys_path, encoding="utf-8") as handle:
                    data = json.load(handle)
            except (OSError, json.JSONDecodeError):
                data = {}
            for name in env_names:
                value = (data.get(name) or "").strip()
                if value:
                    return value
        if "DEEPSEEK_API_KEY" in env_names:
            config_path = os.path.join(directory, "deepseek_config.json")
            if os.path.exists(config_path):
                try:
                    with open(config_path, encoding="utf-8") as handle:
                        data = json.load(handle)
                except (OSError, json.JSONDecodeError):
                    continue
                value = (data.get("api_key") or "").strip()
                if value:
                    return value
    return None


def is_transient_error(message: str) -> bool:
    """区分"网络/限流暂时不通"和"账号/额度这类硬错误"。

    断网时应当等网络回来，而不是把一个配对写成 api_error——那等于把环境故障
    记成模型答错了。余额不足、key 失效属于硬错误，重试再久也没用。
    """
    text = message.lower()
    hard = (
        "insufficient balance",
        "invalid api key",
        "authentication",
        "401",
        "402",
        "account deactivated",
    )
    return not any(token in text for token in hard)


def call_model(client, model: str, messages: List[dict]) -> Tuple[str, int, int, str, int]:
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.0,
        max_tokens=MAX_OUTPUT_TOKENS,
    )
    choice = response.choices[0]
    text = choice.message.content or ""
    usage = response.usage
    prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
    completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
    reasoning_tokens = 0
    details = getattr(usage, "completion_tokens_details", None)
    if details is not None:
        reasoning_tokens = int(getattr(details, "reasoning_tokens", 0) or 0)
    return text, prompt_tokens, completion_tokens, str(choice.finish_reason), reasoning_tokens


def run_task(
    client,
    model: str,
    task: "llm_tasks.Task",
    rounds: int,
    rep: int,
    arm: str = "feedback",
) -> List[dict]:
    """跑一次「任务 × 重复」，返回逐轮记录。

    arm 决定失败之后做什么，这是三个实验组的唯一区别：

    * ``oneshot``：只问一次（等价于 rounds=1）；
    * ``resample``：换一次全新的会话重问同一个题面，**不告诉模型哪里错了**——
      这是「多次采样取成功」的公平基线，用来区分"闭环的反馈"与"单纯多试几次"；
    * ``feedback``：把三层校验的具体错误回传，让模型改。"""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": task.prompt},
    ]
    if arm == "oneshot":
        rounds = 1
    rows: List[dict] = []
    for round_index in range(1, rounds + 1):
        if arm == "resample" and round_index > 1:
            # 独立重采样：不能带上上一轮的错误信息，否则就不是基线了
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": task.prompt},
            ]
        started = time.time()
        deadline = started + NETWORK_WAIT_SECONDS
        error = ""
        text = ""
        prompt_tokens = completion_tokens = reasoning_tokens = 0
        finish_reason = "error"
        attempt_index = 0
        while True:
            attempt_index += 1
            try:
                text, prompt_tokens, completion_tokens, finish_reason, reasoning_tokens = call_model(
                    client, model, messages
                )
                error = ""
                break
            except Exception as exception:  # 网络/额度问题不打断整个实验
                error = f"{type(exception).__name__}: {exception}"
                text = ""
                if not is_transient_error(error):
                    break
                # 短暂断网时在这里等，而不是把环境故障记成模型失败
                if attempt_index >= API_ATTEMPTS and time.time() >= deadline:
                    break
                time.sleep(API_RETRY_SECONDS * min(attempt_index, 6))
        elapsed = time.time() - started

        if error:
            rows.append(
                {
                    "arm": arm,
                    "task": task.key,
                    "rep": rep,
                    "round": round_index,
                    "layer": "api_error",
                    "passed": 0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "elapsed_s": round(elapsed, 2),
                    "detail": error,
                }
            )
            break

        payload, parse_errors = parse_json(text)
        if payload is None:
            # 推理型模型可能把整个补全预算花在思考上，正文为空——单列一类失败，
            # 否则会被误读成"电路写错了"。
            layer_hint = "truncated" if finish_reason == "length" else "syntax"
            report = {
                "gates": [],
                "syntax": parse_errors if layer_hint == "syntax" else [],
                "semantic": [],
                "numeric": [],
                "passed": False,
                "layer_override": layer_hint,
            }
        else:
            report = validate(payload, task)

        rows.append(
            {
                "arm": arm,
                "task": task.key,
                "rep": rep,
                "round": round_index,
                "layer": report.get("layer_override") or failure_layer(report),
                "passed": int(bool(report["passed"])),
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "reasoning_tokens": reasoning_tokens,
                "finish_reason": finish_reason,
                "elapsed_s": round(elapsed, 2),
                "detail": " | ".join(
                    str(item)
                    for layer in ("syntax", "semantic", "numeric")
                    for item in report[layer]
                )[:400],
            }
        )
        if report["passed"]:
            break

        if text.strip():
            messages.append({"role": "assistant", "content": text})
        messages.append({"role": "user", "content": feedback_text(report)})
    return rows


# --------------------------------------------------------------------------
# 汇总与入口
# --------------------------------------------------------------------------
def summarise(rows: List[dict]) -> List[dict]:
    by_group: Dict[tuple, List[dict]] = {}
    for row in rows:
        by_group.setdefault((row.get("arm", "feedback"), row["task"]), []).append(row)

    summary: List[dict] = []
    for (arm, task_key), task_rows in by_group.items():
        reps = sorted({str(row["rep"]) for row in task_rows})
        one_shot, repaired, rounds_used, tokens = [], [], [], 0
        for rep in reps:
            attempt = [row for row in task_rows if str(row["rep"]) == rep]
            if not attempt:
                continue
            one_shot.append(int(attempt[0]["passed"]))
            repaired.append(int(any(row["passed"] for row in attempt)))
            rounds_used.append(len(attempt))
            tokens += sum(
                int(row["prompt_tokens"]) + int(row["completion_tokens"]) for row in attempt
            )
        summary.append(
            {
                "arm": arm,
                "task": task_key,
                "reps": len(reps),
                "pass_at_1": round(float(np.mean(one_shot)) if one_shot else 0.0, 4),
                "final_pass": round(float(np.mean(repaired)) if repaired else 0.0, 4),
                "mean_rounds": round(float(np.mean(rounds_used)) if rounds_used else 0.0, 3),
                "total_tokens": tokens,
            }
        )
    return summary


def _per_pair(rows: List[dict]) -> Dict[str, Dict[tuple, int]]:
    """把逐轮记录压成 arm → {(task, rep): 是否最终成功}。"""
    pairs: Dict[str, Dict[tuple, int]] = {}
    for row in rows:
        arm = row.get("arm", "feedback")
        key = (row["task"], str(row["rep"]))
        pairs.setdefault(arm, {})
        pairs[arm][key] = max(pairs[arm].get(key, 0), int(row["passed"]))
    return pairs


def mcnemar_exact(b: int, c: int) -> float:
    """配对样本的 McNemar 精确检验（双侧）。b、c 是两个不一致格的计数。"""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(b, c) + 1)) / (2 ** n)
    return float(min(1.0, 2 * tail))


def bootstrap_diff_ci(a: List[int], b: List[int], rounds: int = 4000, seed: int = 20261104):
    """配对成功率的差值 b − a 的 bootstrap 95% 置信区间。"""
    rng = np.random.default_rng(seed)
    diffs = np.asarray(b, dtype=float) - np.asarray(a, dtype=float)
    n = len(diffs)
    if n == 0:
        return (0.0, 0.0)
    samples = rng.choice(diffs, size=(rounds, n), replace=True).mean(axis=1)
    return (float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5)))


ARM_ORDER = ("oneshot", "resample", "feedback")


def compare_arms(rows: List[dict]) -> List[dict]:
    """对 arm 两两做配对比较（同 task、同 rep 才算一对）。

    列的语义固定为「前面的 arm 是基线、后面的 arm 是改进」，这样 Δ 的正负号
    永远指向"加了反馈之后变好了多少"，不会因为字典顺序而反转。"""
    pairs = _per_pair(rows)
    arms = [arm for arm in ARM_ORDER if arm in pairs]
    arms += [arm for arm in pairs if arm not in ARM_ORDER]
    out: List[dict] = []
    for index, baseline_arm in enumerate(arms):
        for treatment_arm in arms[index + 1:]:
            common = sorted(set(pairs[treatment_arm]) & set(pairs[baseline_arm]))
            if not common:
                continue
            a = [pairs[baseline_arm][key] for key in common]
            b = [pairs[treatment_arm][key] for key in common]
            b_only = sum(1 for x, y in zip(a, b) if y == 1 and x == 0)
            a_only = sum(1 for x, y in zip(a, b) if y == 0 and x == 1)
            low, high = bootstrap_diff_ci(a, b)
            out.append(
                {
                    "arm_a": baseline_arm,
                    "arm_b": treatment_arm,
                    "pairs": len(common),
                    "rate_a": round(float(np.mean(a)), 4),
                    "rate_b": round(float(np.mean(b)), 4),
                    "delta": round(float(np.mean(b) - np.mean(a)), 4),
                    "ci_low": round(low, 4),
                    "ci_high": round(high, 4),
                    "b_only": b_only,
                    "a_only": a_only,
                    "mcnemar_p": round(mcnemar_exact(b_only, a_only), 4),
                }
            )
    return out


def _write_csv(path: str, rows: List[dict], fields: Sequence[str]) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})
    return path


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="LLM 生成电路的校验—反馈—修复闭环实验")
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--model", default="deepseek-flash")
    parser.add_argument("--limit", type=int, default=0, help="只跑前 N 个任务（调试用）")
    parser.add_argument("--only", default="", help="只跑指定任务 key，逗号分隔")
    parser.add_argument("--mock", action="store_true", help="不调 API，用于跑通流程")
    parser.add_argument(
        "--arms",
        default="feedback,resample",
        help="逗号分隔：oneshot / resample / feedback，默认跑后两组对照",
    )
    parser.add_argument("--tag", default="", help="输出文件名后缀，便于区分不同模型/轮次")
    parser.add_argument(
        "--base-url",
        default="https://api.deepseek.com",
        help="OpenAI 兼容接口地址；换厂商时改这里（Qwen/OpenAI/Ollama 都支持）",
    )
    parser.add_argument(
        "--api-key-env",
        default="DEEPSEEK_API_KEY",
        help="从哪个环境变量读 key，默认 DEEPSEEK_API_KEY；也可写进项目根的 llm_keys.json",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=16000,
        help="单次生成的最大输出 token；小模型上限低时调小，推理型模型要保持大",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="接着已有的 llm_runs_*.csv 跑：跳过已完成的 (arm, task, rep)，中途断了可以续",
    )
    options = parser.parse_args(argv)

    global MAX_OUTPUT_TOKENS
    MAX_OUTPUT_TOKENS = int(options.max_tokens)

    tasks = list(llm_tasks.TASKS)
    if options.only:
        wanted = {item.strip() for item in options.only.split(",") if item.strip()}
        tasks = [task for task in tasks if task.key in wanted]
    if options.limit:
        tasks = tasks[: options.limit]
    arms = [item.strip() for item in options.arms.split(",") if item.strip()]
    tasks_by_key = {task.key: task for task in tasks}

    client = None
    if not options.mock:
        api_key = load_api_key((options.api_key_env,))
        if not api_key:
            print(
                "找不到 API key：设置环境变量 "
                f"{options.api_key_env}，或在项目根目录写 llm_keys.json "
                f'{{"{options.api_key_env}": "..."}}'
            )
            return 2
        from openai import OpenAI

        client = OpenAI(api_key=api_key, base_url=options.base_url)
        print(f"[model] {options.model} @ {options.base_url}")

    fields = [
        "arm",
        "task",
        "rep",
        "round",
        "layer",
        "passed",
        "prompt_tokens",
        "completion_tokens",
        "reasoning_tokens",
        "finish_reason",
        "elapsed_s",
        "detail",
    ]
    suffix = f"_{options.tag}" if options.tag else ""
    runs_path = os.path.join(RESULTS_DIR, f"llm_runs{suffix}.csv")

    rows: List[dict] = []
    done: set = set()
    if options.resume and os.path.exists(runs_path):
        with open(runs_path, encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        done = {(row["arm"], row["task"], int(row["rep"])) for row in rows}
        print(f"[resume] 已有 {len(rows)} 条记录，已完成 {len(done)} 个 (arm, task, rep)", flush=True)

    for arm in arms:
        for task in tasks:
            for rep in range(options.reps):
                if (arm, task.key, rep) in done:
                    continue
                if options.mock:
                    attempt = [
                        {
                            "arm": arm,
                            "task": task.key,
                            "rep": rep,
                            "round": 1,
                            "layer": "pass" if rep % 2 == 0 else "numeric",
                            "passed": int(rep % 2 == 0),
                            "prompt_tokens": 120,
                            "completion_tokens": 90,
                            "reasoning_tokens": 0,
                            "finish_reason": "stop",
                            "elapsed_s": 0.0,
                            "detail": "",
                        }
                    ]
                else:
                    attempt = run_task(client, options.model, task, options.rounds, rep, arm=arm)
                rows.extend(attempt)
                last = attempt[-1]
                print(
                    f"[{arm}][{task.key} rep{rep}] rounds={len(attempt)} "
                    f"passed={bool(last['passed'])} layer={last['layer']}",
                    flush=True,
                )
                _write_csv(runs_path, rows, fields)

    # 断网/限流可能让若干配对以 api_error 收尾。网络若已恢复，这里自动补跑，
    # 避免把环境故障留在数据里。
    if not options.mock:
        for repair_index in range(REPAIR_ROUNDS):
            last_by_pair: Dict[tuple, dict] = {}
            for row in rows:
                last_by_pair[(row["arm"], row["task"], int(row["rep"]))] = row
            failed = {key for key, row in last_by_pair.items() if row["layer"] == "api_error"}
            if not failed:
                break
            print(
                f"[repair {repair_index + 1}/{REPAIR_ROUNDS}] {len(failed)} 个配对以 api_error 收尾，"
                f"{REPAIR_WAIT_SECONDS:.0f}s 后补跑",
                flush=True,
            )
            time.sleep(REPAIR_WAIT_SECONDS)
            rows = [
                row
                for row in rows
                if (row["arm"], row["task"], int(row["rep"])) not in failed
            ]
            for arm, task_key, rep in sorted(failed):
                task = tasks_by_key[task_key]
                attempt = run_task(client, options.model, task, options.rounds, rep, arm=arm)
                rows.extend(attempt)
                last = attempt[-1]
                print(
                    f"[repair][{arm}][{task_key} rep{rep}] rounds={len(attempt)} "
                    f"passed={bool(last['passed'])} layer={last['layer']}",
                    flush=True,
                )
                _write_csv(runs_path, rows, fields)

    summary = summarise(rows)
    summary_path = _write_csv(
        os.path.join(RESULTS_DIR, f"llm_summary{suffix}.csv"),
        summary,
        ["arm", "task", "reps", "pass_at_1", "final_pass", "mean_rounds", "total_tokens"],
    )
    stats = compare_arms(rows)
    stats_path = _write_csv(
        os.path.join(RESULTS_DIR, f"llm_arms{suffix}.csv"),
        stats,
        [
            "arm_a",
            "arm_b",
            "pairs",
            "rate_a",
            "rate_b",
            "delta",
            "ci_low",
            "ci_high",
            "b_only",
            "a_only",
            "mcnemar_p",
        ],
    )

    print(f"\n[ok] {os.path.relpath(runs_path)}")
    print(f"[ok] {os.path.relpath(summary_path)}")
    print(f"[ok] {os.path.relpath(stats_path)}")
    for arm in arms:
        arm_rows = [row for row in summary if row["arm"] == arm]
        if not arm_rows:
            continue
        pass1 = float(np.mean([row["pass_at_1"] for row in arm_rows]))
        final = float(np.mean([row["final_pass"] for row in arm_rows]))
        rounds = float(np.mean([row["mean_rounds"] for row in arm_rows]))
        tokens = int(sum(row["total_tokens"] for row in arm_rows))
        print(
            f"{arm:9s} pass@1 = {pass1:.3f}   final = {final:.3f}   "
            f"平均轮数 = {rounds:.2f}   总 tokens = {tokens}"
        )
    for row in stats:
        print(
            f"[{row['arm_a']} vs {row['arm_b']}] {row['rate_a']:.3f} → {row['rate_b']:.3f} "
            f"(Δ={row['delta']:+.3f}, 95% CI [{row['ci_low']:+.3f}, {row['ci_high']:+.3f}], "
            f"McNemar p={row['mcnemar_p']})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
