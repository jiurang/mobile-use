"""视频分析：把视频文件交给 Gemini 系模型做分析。

视频先按 API 的体积上限压缩，再作为 base64 附件随提示词一起发过去。模型回的
文字就是分析结果；提示词本身在 ``video_analyzer.md`` 和 ``human.md`` 两个模板
里，这里只负责组装消息和清理临时文件。
"""

import base64
from pathlib import Path

from jinja2 import Template
from langchain_core.messages import HumanMessage, SystemMessage

from minitap.mobile_use.context import MobileUseContext
from minitap.mobile_use.services.llm import get_llm, invoke_llm_with_timeout_message, with_fallback
from minitap.mobile_use.utils.logger import get_logger
from minitap.mobile_use.utils.video import compress_video_for_api

logger = get_logger(__name__)


async def analyze_video(
    ctx: MobileUseContext,
    video_path: Path,
    prompt: str,
) -> str:
    """分析一个视频文件，返回模型写的文字说明。

    ``prompt`` 是要问模型的问题，原样拼进 ``human.md`` 模板。视频体积超过 API
    上限时先压缩，压缩产物在函数结束时删掉，原始视频不动。

    Raises:
        FileNotFoundError: 视频文件不存在。
        Exception: 模型调用失败时原样抛出，调用方自己决定要不要重试。
    """
    logger.info(f"Starting video analysis for {video_path}")

    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    # 视频超过 API 的体积上限时先压一份，压缩产物用完即删。
    compressed_path = await compress_video_for_api(video_path)

    try:
        with open(compressed_path, "rb") as video_file:
            video_bytes = video_file.read()

        video_base64 = base64.b64encode(video_bytes).decode("utf-8")

        suffix = compressed_path.suffix.lower()
        mime_type = "video/mp4" if suffix in (".mp4", ".m4v") else f"video/{suffix[1:]}"

        system_message_content = Template(
            Path(__file__).parent.joinpath("video_analyzer.md").read_text(encoding="utf-8")
        ).render()

        human_message_content = Template(
            Path(__file__).parent.joinpath("human.md").read_text(encoding="utf-8")
        ).render(prompt=prompt)

        messages = [
            SystemMessage(content=system_message_content),
            HumanMessage(
                content=[
                    {
                        "type": "text",
                        "text": human_message_content,
                    },
                    {
                        "type": "file",
                        "source_type": "base64",
                        "mime_type": mime_type,
                        "data": video_base64,
                    },
                ]
            ),
        ]

        llm = get_llm(ctx=ctx, name="video_analyzer", is_utils=True, temperature=0.2)
        llm_fallback = get_llm(
            ctx=ctx, name="video_analyzer", is_utils=True, use_fallback=True, temperature=0.2
        )

        logger.info("Sending video to LLM for analysis...")

        response = await with_fallback(
            main_call=lambda: invoke_llm_with_timeout_message(
                llm.ainvoke(messages),
                timeout_seconds=120,
                agent_name="video_analyzer",
                task_id=ctx.trace_id,
            ),
            fallback_call=lambda: invoke_llm_with_timeout_message(
                llm_fallback.ainvoke(messages),
                timeout_seconds=120,
                agent_name="video_analyzer",
                task_id=ctx.trace_id,
                used_fallback=True,
            ),
        )

        content = response.content if hasattr(response, "content") else str(response)
        result = content if isinstance(content, str) else str(content)
        logger.info("Video analysis completed")

        return result
    finally:
        # 只清压缩产物，原始视频不动。
        if compressed_path != video_path and compressed_path.exists():
            try:
                compressed_path.unlink()
            except OSError as e:
                # 清理失败不该影响已经拿到的分析结果，只记一条，别把
                # 「结果成功但临时文件没删掉」变成一次失败。
                logger.debug(f"Could not remove the compressed video: {e}")
