"""Step 2 的三类合成 workload 生成器。

这个文件负责把 `configs/workload.yaml` 里的配置，转换成一组标准化的
`Request` 对象，供后续 benchmark runner 直接发送。

设计目标：

1. 三类 workload 都走统一入口 `generate_workload()`。
2. 所有长度控制尽量通过 tokenizer 的 encode/decode 完成，而不是按字符数截断。
3. RAG workload 的共享前缀要在 token 级别构造，方便后续做 prefix-locality 实验。
4. 每个辅助函数都保持职责单一，便于后续补单元测试。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, is_dataclass
import math
import random
from typing import Any, Protocol, cast

from src.workload.schema import ChatMessage, Request, RequestMetadata, WorkloadType


VLLM_PREFIX_BLOCK_SIZE = 16


class TokenizerLike(Protocol):
    """生成器依赖的最小 tokenizer 协议。

    这里不强绑定 transformers 的具体类型，只要求支持最基本的
    `encode` / `decode`，以及在可用时支持 `apply_chat_template`。
    """

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        ...

    def decode(self, token_ids: Sequence[int], skip_special_tokens: bool = False) -> str:
        ...


CHAT_SYSTEM_PROMPT = (
    "你是一名简洁、可靠的 AI 助手。请直接回答问题，优先给出清晰结论，"
    "必要时再补充简短解释。"
)

RAG_SYSTEM_PROMPT = (
    "你是一名检索增强问答助手。请严格基于提供的资料片段回答问题，"
    "若资料不足，请明确说明证据不足。"
)

AGENT_SYSTEM_PROMPT = (
    "你是一名善于拆解复杂任务的执行型助手。请按步骤分析需求、列出方案，"
    "并给出结构化的最终输出。"
)

CHAT_PROMPT_TEMPLATES = [
    "请解释下面这个概念，并给一个简短例子：",
    "请把下面这段需求总结成三点重点：",
    "请比较两个方案的优缺点，并说明适用场景：",
    "请帮我把下面的问题拆成可以执行的检查步骤：",
]

RAG_QUERY_TEMPLATES = [
    "结合资料，回答用户问题，并指出最相关的证据片段。",
    "根据上下文说明问题的根因、影响范围和建议动作。",
    "从资料中提取关键事实，并按时间线整理结论。",
    "请先概括资料内容，再回答最后的问题。",
]

AGENT_TASK_TEMPLATES = [
    "你需要规划一次线上服务故障排查，输出详细步骤、风险点和回滚建议。",
    "请设计一个实验方案，比较两个系统策略在延迟和吞吐上的差异。",
    "请把一个模糊需求拆成里程碑、依赖项、验收标准和潜在阻塞点。",
    "请围绕一段技术设计说明生成实施计划、测试计划和观测指标。",
]

CORPUS_SNIPPETS = [
    "大型语言模型在线服务需要同时平衡吞吐、交互延迟和资源利用率。",
    "在多实例部署场景里，请求路由策略会直接影响 prefix cache 的局部性。",
    "如果请求共享长前缀，被调度到同一个实例通常更容易命中已有缓存。",
    "实验平台的关键不只是跑通请求，还包括统计口径稳定、结果可复现。",
    "真实系统中的混合负载往往同时包含短问答、长上下文检索和长输出任务。",
    "为了让 benchmark 结果可信，我们通常需要固定 warmup、样本量和输出格式。",
    "同一套 workload schema 可以降低 generator、runner 和 metrics 之间的耦合。",
    "prefix-aware routing 的价值取决于共享前缀比例、请求长度和负载不均衡程度。",
    "当输入 prompt 很长时，TTFT 往往会显著受到 prefill 阶段成本的影响。",
    "长输出任务通常会拉高端到端延迟，因此需要单独关注 TPOT 和吞吐指标。",
]


def generate_workload(
    config: Mapping[str, Any] | Any,
    tokenizer: TokenizerLike,
    seed: int,
) -> list[Request]:
    """生成完整 workload 请求列表。

    参数说明：
    - `config` 可以是字典，也可以是后续 runner 可能传入的 dataclass / 配置对象。
    - `tokenizer` 用来做 token 级长度控制和 RAG 共享前缀构造。
    - `seed` 控制整个生成过程的可复现性。

    返回值：
    - 一个 `Request` 列表；arrival offset 先统一置为 0，后续由 arrival 模块补上。
    """

    workload_cfg = _unwrap_workload_config(config)
    rng = random.Random(seed)

    num_requests = int(workload_cfg["num_requests"])
    mix = workload_cfg["mix"]
    chat_cfg = workload_cfg["chat"]
    rag_cfg = workload_cfg["rag"]
    agent_cfg = workload_cfg["agent"]

    request_types = _sample_workload_types(num_requests=num_requests, mix=mix, rng=rng)
    rag_prefix_state = _build_rag_prefix_state(rag_cfg=rag_cfg, tokenizer=tokenizer, rng=rng)

    requests: list[Request] = []
    for index, workload_type in enumerate(request_types):
        request_seed = rng.randint(0, 10**9)
        request_rng = random.Random(request_seed)

        if workload_type == "chat":
            request = _generate_chat_request(
                request_index=index,
                cfg=chat_cfg,
                tokenizer=tokenizer,
                rng=request_rng,
                seed=request_seed,
            )
        elif workload_type == "rag":
            request = _generate_rag_request(
                request_index=index,
                cfg=rag_cfg,
                tokenizer=tokenizer,
                rng=request_rng,
                seed=request_seed,
                prefix_state=rag_prefix_state,
            )
        else:
            request = _generate_agent_request(
                request_index=index,
                cfg=agent_cfg,
                tokenizer=tokenizer,
                rng=request_rng,
                seed=request_seed,
            )

        requests.append(request)

    return requests


def render_messages_to_token_ids(
    tokenizer: TokenizerLike,
    messages: Sequence[Mapping[str, Any]],
) -> list[int]:
    """把 OpenAI chat messages 渲染成 token id 列表。

    优先使用 tokenizer 自带的 chat template，这样结果最接近模型真实收到的输入。
    如果 tokenizer 不支持 chat template，就退化为一个稳定的文本拼接格式。
    """

    apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
    if callable(apply_chat_template):
        token_ids = cast(
            Sequence[int],
            apply_chat_template(
            list(messages),
            tokenize=True,
            add_generation_prompt=True,
            ),
        )
        return list(token_ids)

    fallback_text = []
    for message in messages:
        role = message.get("role", "user")
        content = message.get("content", "")
        fallback_text.append(f"<{role}>\n{content}\n</{role}>")
    fallback_text.append("<assistant>\n")
    return list(tokenizer.encode("\n".join(fallback_text), add_special_tokens=False))


def _unwrap_workload_config(config: Mapping[str, Any] | Any) -> Mapping[str, Any]:
    """把不同形态的配置对象统一转换成字典视图。"""

    if isinstance(config, Mapping):
        if "workload" in config and isinstance(config["workload"], Mapping):
            return config["workload"]
        return config

    if is_dataclass(config) and not isinstance(config, type):
        data = cast(dict[str, Any], asdict(cast(Any, config)))
        if "workload" in data and isinstance(data["workload"], Mapping):
            return data["workload"]
        return data

    if hasattr(config, "workload"):
        nested = getattr(config, "workload")
        return _unwrap_workload_config(nested)

    if hasattr(config, "__dict__"):
        return dict(vars(config))

    raise TypeError("unsupported workload config type")


def _sample_workload_types(
    *,
    num_requests: int,
    mix: Mapping[str, Any],
    rng: random.Random,
) -> list[WorkloadType]:
    """按 mix 比例抽样 workload 类型。"""

    weights = {
        "chat": float(mix["chat"]),
        "rag": float(mix["rag"]),
        "agent": float(mix["agent"]),
    }
    total = sum(weights.values())
    if total <= 0:
        raise ValueError("workload mix weights must sum to a positive value")

    population: list[WorkloadType] = ["chat", "rag", "agent"]
    normalized = [weights[item] / total for item in population]
    return rng.choices(population, weights=normalized, k=num_requests)


def _generate_chat_request(
    *,
    request_index: int,
    cfg: Mapping[str, Any],
    tokenizer: TokenizerLike,
    rng: random.Random,
    seed: int,
) -> Request:
    """生成 chat workload 请求。

    chat 请求的特点是：
    - prompt 相对较短
    - 不共享前缀
    - 输出长度中等偏短
    """

    prompt_target = _randint_from_cfg(cfg, "prompt_tokens_min", "prompt_tokens_max", rng)
    max_tokens = _randint_from_cfg(cfg, "max_tokens_min", "max_tokens_max", rng)
    min_tokens = _optional_int(cfg.get("min_tokens"))

    instruction = rng.choice(CHAT_PROMPT_TEMPLATES)
    unique_body = _build_segment_text(
        tokenizer=tokenizer,
        rng=rng,
        target_tokens=max(64, prompt_target * 2),
        context_label="chat",
    )
    messages, prompt_actual = _fit_messages_to_target(
        tokenizer=tokenizer,
        system_prompt=CHAT_SYSTEM_PROMPT,
        user_prefix=f"{instruction}\n\n",
        variable_suffix=unique_body,
        target_total_tokens=prompt_target,
    )

    return _build_request(
        request_index=request_index,
        workload_type="chat",
        messages=messages,
        prompt_target=prompt_target,
        prompt_actual=prompt_actual,
        max_tokens=max_tokens,
        min_tokens=min_tokens,
        prefix_group_id=None,
        shared_prefix_tokens=0,
        seed=seed,
    )


def _generate_rag_request(
    *,
    request_index: int,
    cfg: Mapping[str, Any],
    tokenizer: TokenizerLike,
    rng: random.Random,
    seed: int,
    prefix_state: Mapping[int, dict[str, Any]],
) -> Request:
    """生成 RAG workload 请求。

    这里的重点不是“像不像真实知识库文本”，而是：
    - 共享前缀是否稳定
    - token 级长度控制是否成立
    - 同组请求是否在完整 prompt 的前缀上保持一致
    """

    prompt_target = _randint_from_cfg(cfg, "prompt_tokens_min", "prompt_tokens_max", rng)
    max_tokens = _randint_from_cfg(cfg, "max_tokens_min", "max_tokens_max", rng)
    min_tokens = _optional_int(cfg.get("min_tokens"))

    shared_ratio = float(cfg["shared_prefix_ratio"])
    use_shared_prefix = rng.random() < shared_ratio

    prefix_group_id: int | None = None
    prefix_text = ""
    shared_prefix_tokens = 0
    reference_prompt_prefix_ids: list[int] | None = None

    if use_shared_prefix:
        prefix_group_id = rng.choice(list(prefix_state.keys()))
        prefix_payload = prefix_state[prefix_group_id]
        prefix_text = prefix_payload["prefix_text"]
        shared_prefix_tokens = int(prefix_payload["shared_prefix_tokens"])
        reference_prompt_prefix_ids = list(
            prefix_payload["rendered_prefix_token_ids"][:shared_prefix_tokens]
        )
    else:
        unique_prefix_tokens = _aligned_block_token_count(
            int(cfg["shared_prefix_tokens"]),
            VLLM_PREFIX_BLOCK_SIZE,
        )
        unique_prefix_text = _build_segment_text(
            tokenizer=tokenizer,
            rng=rng,
            target_tokens=max(unique_prefix_tokens, VLLM_PREFIX_BLOCK_SIZE),
            context_label="rag_private_prefix",
        )
        prefix_text = (
            "资料片段（本请求独有）:\n"
            f"{unique_prefix_text}\n\n"
            "请结合上述资料回答最后的问题。\n\n"
        )

    rag_query = _compose_rag_query(rng=rng, tokenizer=tokenizer, prompt_target=prompt_target)
    messages, prompt_actual = _fit_messages_to_target(
        tokenizer=tokenizer,
        system_prompt=RAG_SYSTEM_PROMPT,
        user_prefix=prefix_text,
        variable_suffix=rag_query,
        target_total_tokens=prompt_target,
    )

    if reference_prompt_prefix_ids is not None:
        full_prompt_ids = render_messages_to_token_ids(tokenizer, messages)
        if full_prompt_ids[: len(reference_prompt_prefix_ids)] != reference_prompt_prefix_ids:
            raise ValueError("shared-prefix RAG request does not preserve group prefix")

    return _build_request(
        request_index=request_index,
        workload_type="rag",
        messages=messages,
        prompt_target=prompt_target,
        prompt_actual=prompt_actual,
        max_tokens=max_tokens,
        min_tokens=min_tokens,
        prefix_group_id=prefix_group_id,
        shared_prefix_tokens=shared_prefix_tokens,
        seed=seed,
    )


def _generate_agent_request(
    *,
    request_index: int,
    cfg: Mapping[str, Any],
    tokenizer: TokenizerLike,
    rng: random.Random,
    seed: int,
) -> Request:
    """生成 agent workload 请求。

    Step 2 的 agent workload 不模拟真实 tool calling，只模拟：
    - prompt 比 chat 更长
    - 输出比 chat/RAG 更长
    - 任务描述往往更复杂、更结构化
    """

    prompt_target = _randint_from_cfg(cfg, "prompt_tokens_min", "prompt_tokens_max", rng)
    max_tokens = _randint_from_cfg(cfg, "max_tokens_min", "max_tokens_max", rng)
    configured_min = _optional_int(cfg.get("min_tokens"))
    min_tokens = configured_min if configured_min is not None else max(max_tokens - 50, 1)
    min_tokens = min(min_tokens, max_tokens)

    template_count = int(cfg.get("num_task_templates", len(AGENT_TASK_TEMPLATES)))
    instruction = rng.choice(AGENT_TASK_TEMPLATES[:template_count])
    planning_context = _build_segment_text(
        tokenizer=tokenizer,
        rng=rng,
        target_tokens=max(160, prompt_target * 2),
        context_label="agent",
    )
    messages, prompt_actual = _fit_messages_to_target(
        tokenizer=tokenizer,
        system_prompt=AGENT_SYSTEM_PROMPT,
        user_prefix=f"{instruction}\n\n任务上下文：\n",
        variable_suffix=planning_context,
        target_total_tokens=prompt_target,
    )

    return _build_request(
        request_index=request_index,
        workload_type="agent",
        messages=messages,
        prompt_target=prompt_target,
        prompt_actual=prompt_actual,
        max_tokens=max_tokens,
        min_tokens=min_tokens,
        prefix_group_id=None,
        shared_prefix_tokens=0,
        seed=seed,
    )


def _build_rag_prefix_state(
    *,
    rag_cfg: Mapping[str, Any],
    tokenizer: TokenizerLike,
    rng: random.Random,
) -> dict[int, dict[str, Any]]:
    """预先构造所有共享前缀组。

    这样后续每个 RAG 请求只需要按 group_id 取对应前缀即可，不会在运行时
    临时重新生成，保证同组请求的共享部分完全一致。
    """

    num_groups = int(rag_cfg["num_prefix_groups"])
    raw_shared_prefix_tokens = int(rag_cfg["shared_prefix_tokens"])
    aligned_shared_prefix_tokens = _aligned_block_token_count(
        raw_shared_prefix_tokens,
        VLLM_PREFIX_BLOCK_SIZE,
    )

    state: dict[int, dict[str, Any]] = {}
    for group_id in range(num_groups):
        base_segment = _build_segment_text(
            tokenizer=tokenizer,
            rng=rng,
            target_tokens=max(aligned_shared_prefix_tokens * 2, 256),
            context_label=f"rag_group_{group_id}",
        )
        prefix_wrapper = (
            f"资料组 {group_id}：\n"
            f"{base_segment}\n\n"
            "以上资料可能包含背景、配置、日志片段和约束条件。\n"
        )
        prefix_token_ids = _text_to_exact_token_ids(
            tokenizer=tokenizer,
            text=prefix_wrapper,
            target_tokens=aligned_shared_prefix_tokens,
        )
        prefix_text = (
            tokenizer.decode(prefix_token_ids, skip_special_tokens=False)
            + "\n\n以上是共享资料前缀。下面开始给出本次请求的具体问题。\n\n"
        )

        rendered_prefix_token_ids = _render_stable_prompt_prefix_ids(
            tokenizer=tokenizer,
            system_prompt=RAG_SYSTEM_PROMPT,
            user_prefix=prefix_text,
        )

        if len(rendered_prefix_token_ids) < aligned_shared_prefix_tokens:
            raise ValueError("rendered RAG shared prefix is shorter than requested token length")

        state[group_id] = {
            "prefix_text": prefix_text,
            "shared_prefix_tokens": aligned_shared_prefix_tokens,
            "rendered_prefix_token_ids": rendered_prefix_token_ids,
        }

    return state


def _render_stable_prompt_prefix_ids(
    *,
    tokenizer: TokenizerLike,
    system_prompt: str,
    user_prefix: str,
) -> list[int]:
    """提取“真正稳定”的完整 prompt 前缀 token。

    直接把“只有 prefix 的消息”拿来当参考会有一个问题：
    那个结果会包含 chat template 的收尾 token，例如 `</user>` 或 assistant 前缀；
    但真实请求在 prefix 后面还会继续拼 query，因此两者在 prefix 末尾处分叉。

    这里采用一个更稳的方法：
    - 渲染一次“只有 prefix”的消息
    - 渲染一次“prefix + 占位后缀”的消息
    - 取两者的最长公共前缀

    这样留下来的 token 才是“无论后面接什么 query 都不变”的稳定前缀。
    """

    base_ids = render_messages_to_token_ids(
        tokenizer,
        _make_messages(system_prompt=system_prompt, user_content=user_prefix),
    )
    extended_ids = render_messages_to_token_ids(
        tokenizer,
        _make_messages(system_prompt=system_prompt, user_content=user_prefix + "\n\n占位查询"),
    )

    common_prefix: list[int] = []
    for left, right in zip(base_ids, extended_ids):
        if left != right:
            break
        common_prefix.append(left)
    return common_prefix


def _fit_messages_to_target(
    *,
    tokenizer: TokenizerLike,
    system_prompt: str,
    user_prefix: str,
    variable_suffix: str,
    target_total_tokens: int,
) -> tuple[list[ChatMessage], int]:
    """把 messages 调整到尽量接近目标 token 数。

    这里的核心思路是：
    1. 固定 system prompt 和 user prefix。
    2. 只通过裁剪或扩展 `variable_suffix` 来逼近目标长度。
    3. 每轮都重新按完整 chat messages 计算 token 数，而不是只算 user 文本。

    这样实际记录下来的 `prompt_tokens_actual` 才更接近模型真实收到的 prompt。
    """

    suffix_ids = list(tokenizer.encode(variable_suffix, add_special_tokens=False))
    if not suffix_ids:
        suffix_ids = list(tokenizer.encode("补充说明。", add_special_tokens=False))

    messages = _make_messages(
        system_prompt=system_prompt,
        user_content=user_prefix + tokenizer.decode(suffix_ids, skip_special_tokens=False),
    )
    actual = len(render_messages_to_token_ids(tokenizer, messages))

    for _ in range(12):
        diff = target_total_tokens - actual
        if diff == 0:
            break

        if diff > 0:
            extension_ids = suffix_ids[: min(len(suffix_ids), diff)]
            if not extension_ids:
                extension_ids = suffix_ids[:1]
            suffix_ids.extend(extension_ids)
        else:
            trim = min(len(suffix_ids) - 1, abs(diff))
            if trim <= 0:
                break
            suffix_ids = suffix_ids[:-trim]

        messages = _make_messages(
            system_prompt=system_prompt,
            user_content=user_prefix + tokenizer.decode(suffix_ids, skip_special_tokens=False),
        )
        actual = len(render_messages_to_token_ids(tokenizer, messages))

    return messages, actual


def _compose_rag_query(
    *,
    rng: random.Random,
    tokenizer: TokenizerLike,
    prompt_target: int,
) -> str:
    """生成每个 RAG 请求独有的 query 部分。"""

    lead = rng.choice(RAG_QUERY_TEMPLATES)
    unique_context = _build_segment_text(
        tokenizer=tokenizer,
        rng=rng,
        target_tokens=max(96, math.ceil(prompt_target * 0.4)),
        context_label="rag_query",
    )
    return (
        "用户问题：\n"
        f"{lead}\n\n"
        "补充上下文：\n"
        f"{unique_context}\n\n"
        "请输出：结论、证据、风险。"
    )


def _build_segment_text(
    *,
    tokenizer: TokenizerLike,
    rng: random.Random,
    target_tokens: int,
    context_label: str,
) -> str:
    """构造一段足够长的基础文本，再按 token 长度截取。

    这里不依赖外部数据集，而是使用一组稳定模板反复拼接，保证：
    - 本地离线环境可运行
    - 每次生成都有足够多 token 可供切片
    - 不同 workload 之间仍有一定文本差异
    """

    target_tokens = max(target_tokens, 16)
    fragments: list[str] = [f"[{context_label}]"]
    while len(tokenizer.encode(" ".join(fragments), add_special_tokens=False)) < target_tokens * 2:
        fragments.append(rng.choice(CORPUS_SNIPPETS))
    text = " ".join(fragments)
    token_ids = _text_to_exact_token_ids(
        tokenizer=tokenizer,
        text=text,
        target_tokens=target_tokens,
    )
    return tokenizer.decode(token_ids, skip_special_tokens=False)


def _text_to_exact_token_ids(
    *,
    tokenizer: TokenizerLike,
    text: str,
    target_tokens: int,
) -> list[int]:
    """把文本裁剪或扩展到精确的 token 数。"""

    if target_tokens <= 0:
        raise ValueError("target_tokens must be positive")

    token_ids = list(tokenizer.encode(text, add_special_tokens=False))
    if not token_ids:
        token_ids = list(tokenizer.encode("占位文本。", add_special_tokens=False))

    while len(token_ids) < target_tokens:
        remaining = target_tokens - len(token_ids)
        token_ids.extend(token_ids[:remaining])

    return token_ids[:target_tokens]


def _aligned_block_token_count(raw_tokens: int, block_size: int) -> int:
    """把 token 数向下对齐到 block size 的整数倍。"""

    if raw_tokens <= 0:
        raise ValueError("raw_tokens must be positive")
    aligned = (raw_tokens // block_size) * block_size
    if aligned <= 0:
        raise ValueError("aligned token count must be positive")
    return aligned


def _randint_from_cfg(
    cfg: Mapping[str, Any],
    min_key: str,
    max_key: str,
    rng: random.Random,
) -> int:
    """从配置范围里采样整数，并校验区间合法性。"""

    low = int(cfg[min_key])
    high = int(cfg[max_key])
    if low > high:
        raise ValueError(f"invalid range: {min_key}={low} > {max_key}={high}")
    return rng.randint(low, high)


def _optional_int(value: Any) -> int | None:
    """把可空配置值统一转换成 `int | None`。"""

    if value is None:
        return None
    return int(value)


def _make_messages(*, system_prompt: str, user_content: str) -> list[ChatMessage]:
    """构造统一的双消息对话格式。"""

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]


def _build_request(
    *,
    request_index: int,
    workload_type: WorkloadType,
    messages: list[ChatMessage],
    prompt_target: int,
    prompt_actual: int,
    max_tokens: int,
    min_tokens: int | None,
    prefix_group_id: int | None,
    shared_prefix_tokens: int,
    seed: int,
) -> Request:
    """把已生成的消息和统计字段封装成标准 `Request`。"""

    metadata = RequestMetadata(
        prompt_tokens_target=prompt_target,
        prompt_tokens_actual=prompt_actual,
        prefix_group_id=prefix_group_id,
        shared_prefix_tokens=shared_prefix_tokens,
        arrival_time_offset=0.0,
        seed=seed,
    )

    return Request(
        request_id=f"req-{request_index:06d}",
        workload_type=workload_type,
        messages=messages,
        max_tokens=max_tokens,
        min_tokens=min_tokens,
        temperature=0.0,
        metadata=metadata,
    )


__all__ = [
    "TokenizerLike",
    "VLLM_PREFIX_BLOCK_SIZE",
    "generate_workload",
    "render_messages_to_token_ids",
]
