"""提示词语言切换（中 / 英）+ 提示词文件加载。

**为什么需要它**
--------------
提示词是这个系统里**唯一**决定模型行为的部分，而中文任务的界面、目标、
思考全是中文 —— 让模型读中文提示词，不是为了省钱（实测中文反而每轮多约
900 token，见 ``doc/prompt-i18n-ab.md``），而是为了**让模型的「思考」和
用户的语言对齐**：模型的 `agent_thought` 也就是给人看的执行过程。

**开关在哪三层**
----------------
1. **任务级** —— ``TaskRequest.prompt_language``（SDK: ``task.with_prompt_language(...)``，
   CLI: ``--prompt-language``）。**优先级最高**：同一个 Agent 上跑两个任务，
   一个中文一个英文，天然成立。
2. **Agent 级** —— ``AgentConfig.prompt_language``（SDK: ``config.with_prompt_language(...)``）。
   这个 Agent 的默认语言。
3. **环境变量** —— ``MOBILE_USE_PROMPT_LANGUAGE``。不改代码就能切，
   方便真机 A/B（跑一次中文、改回去再跑一次英文）。

优先级：**任务级 > Agent 级 > 环境变量 > 默认英文**。

**缺中文版怎么办**
----------------
不是每个提示词都有中文版（``planner.md`` / ``video_analyzer.md`` 目前还没有）。
缺的时候**回落英文 + 明确 warning**，绝不硬拦：提示词是「跑起来并上报事实」
的地方，拦下来只会让整轮任务挂掉，而「以为在跑中文、其实跑的是英文」
这件事必须**说出来**，否则 A/B 的结论是错的。

**文件名约定**
--------------
中文版 = 英文文件名 + 后缀 ``-zh``（``contextor.md`` → ``contextor-zh.md``）。
历史上 ``cortex.zh.md`` 用的是点号，所以两种后缀都认（见 :data:`ZH_SUFFIXES`）——
不认的话，那个文件会因为「找不到中文版」被静默回落成英文。
"""

from __future__ import annotations

import os
from enum import StrEnum
from functools import cache
from pathlib import Path

from jinja2 import Template

from minitap.mobile_use.utils.logger import get_logger

logger = get_logger(__name__)


class PromptLanguage(StrEnum):
    """提示词语言。``en`` = 英文原版，``zh`` = 中文版。

    ⚠️ **别把它改成 ``Literal["en", "zh"]``** —— 那样 CLI 的
    ``--prompt-language`` 就没了：Typer **不支持** ``Literal``（会抛
    ``Type not yet supported: typing.Literal['en', 'zh']``），只有 Enum 能
    渲染出 ``[en|zh]`` 的选择列表。

    顺带的好处：pydantic 能把 ``"zh"`` 直接收成 ``PromptLanguage.ZH``，
    而 StrEnum 本身就是 ``str``，所以 ``language == "zh"`` 依然成立。
    """

    EN = "en"
    ZH = "zh"


#: 默认语言。不设任何开关时的行为 = 今天的逐字节行为（英文）。
DEFAULT_PROMPT_LANGUAGE: PromptLanguage = PromptLanguage.EN

#: 环境变量名。不改代码就能切语言（真机 A/B 用）。
PROMPT_LANGUAGE_ENV = "MOBILE_USE_PROMPT_LANGUAGE"

#: 中文版文件名的后缀候选，按优先级排。
#:
#: * ``-zh`` —— **现行约定**（``00-identity-zh.md``、本次新增的一批）。
#: * ``.zh`` —— 历史遗留（``cortex.zh.md``）。不认它，那个文件会被静默回落成英文。
ZH_SUFFIXES: tuple[str, ...] = ("-zh", ".zh")

#: 各种写法的归一化表。用户写 ``zh-CN`` / ``Chinese`` / ``中文`` 都认。
_LANGUAGE_ALIASES: dict[str, PromptLanguage] = {
    "en": PromptLanguage.EN,
    "en-us": PromptLanguage.EN,
    "en_us": PromptLanguage.EN,
    "en-gb": PromptLanguage.EN,
    "english": PromptLanguage.EN,
    "英文": PromptLanguage.EN,
    "zh": PromptLanguage.ZH,
    "zh-cn": PromptLanguage.ZH,
    "zh_cn": PromptLanguage.ZH,
    "zh-hans": PromptLanguage.ZH,
    "cn": PromptLanguage.ZH,
    "chinese": PromptLanguage.ZH,
    "中文": PromptLanguage.ZH,
}


def normalize_language(value: str | None) -> PromptLanguage | None:
    """把字符串收成合法语言。**认不出来返回 None**（= 没指定，不是「英文」）。

    分清「没指定」和「指定了英文」很重要：前者要继续往下找（agent 级 →
    环境变量），后者直接定案。

    ``None`` 也不等于「英文」—— 那是调用方在回答「你指定语言了吗」。
    """
    if value is None:
        return None
    return _LANGUAGE_ALIASES.get(value.strip().lower())


def coerce_prompt_language(value: object) -> PromptLanguage | None:
    """把**任意**输入收成合法语言，供 pydantic 字段校验用。

    与 :func:`normalize_language` 的区别：这里接受 ``object``（字段校验器拿到的
    东西类型不定），并且**认不出来时出声**。

    ⚠️ **认不出来返回 None，不抛错** —— 一个写错的值不该让整轮任务挂掉。
    回落英文 + 上报即可：提示词语言是「跑起来并上报事实」的地方，
    硬拦只会把「语言配错了」升级成「任务跑不了」。
    """
    if value is None or isinstance(value, PromptLanguage):
        return value
    if not isinstance(value, str):
        logger.warning(f"[prompt-language] 语言 {value!r} 不是字符串，按未指定处理")
        return None
    language = _LANGUAGE_ALIASES.get(value.strip().lower())
    if language is None:
        logger.warning(f"[prompt-language] 语言 {value!r} 认不出来，按未指定处理")
    return language


def resolve_prompt_language(
    *,
    task: str | None = None,
    agent: str | None = None,
) -> PromptLanguage:
    """定出这次运行用哪种语言。

    优先级：**任务级 > Agent 级 > 环境变量 > 默认英文**。

    非法值（环境变量写了 ``klingon``、SDK 传了 ``"zh_CN_extra"``）不抛错 ——
    只 warning 并继续往下找。一个写错的环境变量不该让整轮任务挂掉。
    """
    sources: tuple[tuple[str, str | None], ...] = (
        ("task 级", task),
        ("agent 级", agent),
        (f"环境变量 {PROMPT_LANGUAGE_ENV}", os.getenv(PROMPT_LANGUAGE_ENV)),
    )
    for label, value in sources:
        if not value:
            continue
        language = normalize_language(value)
        if language is not None:
            return language
        logger.warning(f"[prompt-language] {label}的值 {value!r} 认不出来，忽略")
    return DEFAULT_PROMPT_LANGUAGE


def resolve_prompt_path(base_path: Path, language: PromptLanguage) -> Path:
    """找到这个提示词在指定语言下对应的文件。

    * ``en`` → 原文件。
    * ``zh`` → 先试 ``<stem>-zh.md``，再试 ``<stem>.zh.md``（历史命名），
      都没有就**回落英文并 warning**。

    回落必须**出声**：「以为在跑中文、其实跑的是英文」会直接把 A/B 结论带歪，
    而这种错误在输出里完全看不出来。
    """
    if language == PromptLanguage.EN:
        return base_path

    for suffix in ZH_SUFFIXES:
        candidate = base_path.with_name(f"{base_path.stem}{suffix}{base_path.suffix}")
        if candidate.is_file():
            return candidate

    tried = ", ".join(repr(f"{base_path.stem}{s}{base_path.suffix}") for s in ZH_SUFFIXES)
    logger.warning(
        f"[prompt-language] 语言=zh 但 {base_path.name!r} 没有中文版"
        f"（试过 {tried}），回落英文 {base_path.name!r}"
    )
    return base_path


def read_prompt(base_path: Path, language: PromptLanguage) -> str:
    """读指定语言下的提示词正文（UTF-8）。"""
    return resolve_prompt_path(base_path, language).read_text(encoding="utf-8")


@cache
def load_prompt_template(base_path: Path, language: PromptLanguage) -> Template:
    """加载并编译提示词模板 —— **所有 agent 取提示词的唯一入口**。

    缓存键是 ``(文件路径, 语言)``：模板编译一次就够，而
    ``resolve_prompt_path`` 里的文件系统探测（每次要 stat 一遍）也就只在
    第一次付。

    ⚠️ **缓存的前提：提示词文件在进程生命周期内不变。** 它们是随包发布的
    静态资源，不是运行期生成物 —— 所以「改了文件要重启」这条约束在这里
    是可接受的（开发时改提示词本来就要重启 agent）。
    """
    return Template(read_prompt(base_path, language))


def pick(en: str, zh: str, *, language: PromptLanguage) -> str:
    """代码里拼的固定文案按语言挑一份。

    给「不在 .md 文件里、但会进 prompt 的那一小段文字」用 —— 比如
    Outputter 的 system 提示、Reporter 的人话标签。

    **英文在前、中文在后，并排写出来** —— 翻译的人一眼看到中英对照，
    不用跳到别处找。``language`` 是配置（要能被安全地加在末尾、
    不受位置影响），所以走**关键字参数**。

    刻意不做成 dict 查表：两种语言并排写出来，对照关系就在眼前。
    """
    return zh if language == PromptLanguage.ZH else en
