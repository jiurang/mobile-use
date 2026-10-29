from unittest.mock import AsyncMock, Mock, patch

import pytest
from pydantic import BaseModel

from minitap.mobile_use.agents.outputter.outputter import outputter
from minitap.mobile_use.config import LLM, OutputConfig
from minitap.mobile_use.context import MobileUseContext
from minitap.mobile_use.prompt_language import PromptLanguage

# 这里不要给 sys.modules 打桩。
#
# 打桩会把假的 ``langchain_openai`` 喂给 ``minitap.mobile_use.services.llm``，
# 而那个模块在 import 期就要继承 ``ChatOpenAI``。假的类一旦被继承，
# ``MiniMaxChatOpenAI`` / ``SurgeryChatOpenAI`` 就变成「metaclass 是 Mock」的假
# 类，假的 llm 模块还会留在 sys.modules 里污染整个 pytest 会话。后来收集到的
# 测试拿到的是假客户端，症状是「单跑这个文件通过，全量跑一片红」：实测全量跑
# 有 36 个测试失败（test_minimax_effort / test_deepseek_provider /
# test_outputter 自己都挂），根因就是这里的打桩。同理，给 llm.py 的工具 schema
# 手术加第二个基类会让 llm.py 抛 ``TypeError: metaclass conflict``，15 个测试
# 模块在收集期就崩。
#
# 现在也不需要打桩：这些 import 不装任何桩也能跑通，可选的 provider 包
#（vertexai / azure / cerebras）只在 TYPE_CHECKING 下引用，运行时根本不 import。


class MockPydanticSchema(BaseModel):
    color: str
    price: float
    currency_symbol: str
    website_url: str


mock_dict = {
    "color": "green",
    "price": 20,
    "currency_symbol": "$",
    "website_url": "http://superwebsite.fr",
}


class DummyState:
    """State 的最小替身：outputter 只读这两个字段。"""

    def __init__(self, initial_goal, agents_thoughts) -> None:
        self.initial_goal = initial_goal
        self.agents_thoughts = agents_thoughts


@pytest.fixture
def mock_context() -> Mock:
    """Create a properly mocked context with all required fields."""
    ctx = Mock(spec=MobileUseContext)
    # pydantic v2 的字段名不在类的 ``dir()`` 里，所以 ``Mock(spec=...)`` 不会
    # 自动生成它们，``ctx.trace_id`` 会抛 AttributeError（真实的 llm 路径要读
    # 它）。这不是生产代码的问题，是 pydantic 模型当 spec 时的固有坑。
    ctx.trace_id = "test-trace-id"
    ctx.llm_config = {
        "executor": LLM(provider="openai", model="gpt-5-nano"),
        "cortex": LLM(provider="openai", model="gpt-5-nano"),
        "planner": LLM(provider="openai", model="gpt-5-nano"),
        "orchestrator": LLM(provider="openai", model="gpt-5-nano"),
    }
    ctx.device = Mock()
    # 提示词语言：outputter 要按它挑 prompts/<lang>/human.md。
    # 同样是 pydantic 字段，Mock(spec=...) 不会自动生成，得显式给。
    ctx.prompt_language = PromptLanguage.EN
    return ctx


@pytest.fixture
def mock_state() -> DummyState:
    """Create a mock state with test data."""
    return DummyState(
        initial_goal="Find a green product on my website",
        agents_thoughts=[
            "Going on http://superwebsite.fr",
            "Searching for products",
            "Filtering by color",
            "Color 'green' found for a 20 dollars product",
        ],
    )


@patch("minitap.mobile_use.agents.outputter.outputter.get_llm")
@pytest.mark.asyncio
async def test_outputter_with_pydantic_model(mock_get_llm, mock_context, mock_state):
    """Test outputter with Pydantic model output."""
    # Mock the structured LLM response
    mock_structured_llm = AsyncMock()
    mock_structured_llm.ainvoke.return_value = MockPydanticSchema(
        color="green", price=20, currency_symbol="$", website_url="http://superwebsite.fr"
    )

    # Mock the base LLM
    mock_llm = Mock()
    mock_llm.with_structured_output.return_value = mock_structured_llm
    mock_get_llm.return_value = mock_llm

    config = OutputConfig(
        structured_output=MockPydanticSchema,
        output_description=None,
    )

    result = await outputter(ctx=mock_context, output_config=config, graph_output=mock_state)

    assert isinstance(result, dict)
    assert result.get("color") == "green"


@patch("minitap.mobile_use.agents.outputter.outputter.get_llm")
@pytest.mark.asyncio
async def test_outputter_with_dict(mock_get_llm, mock_context, mock_state):
    """Test outputter with dictionary output."""
    # Mock the structured LLM response for dict
    mock_structured_llm = AsyncMock()
    expected_dict = {
        "color": "green",
        "price": 20,
        "currency_symbol": "$",
        "website_url": "http://superwebsite.fr",
    }
    mock_structured_llm.ainvoke.return_value = expected_dict

    # Mock the base LLM
    mock_llm = Mock()
    mock_llm.with_structured_output.return_value = mock_structured_llm
    mock_get_llm.return_value = mock_llm

    config = OutputConfig(
        structured_output=mock_dict,
        output_description=None,
    )

    result = await outputter(ctx=mock_context, output_config=config, graph_output=mock_state)

    assert isinstance(result, dict)
    assert result.get("color") == "green"
    assert result.get("price") == 20
    assert result.get("currency_symbol") == "$"
    assert result.get("website_url") == "http://superwebsite.fr"


@patch("minitap.mobile_use.agents.outputter.outputter.get_llm")
@pytest.mark.asyncio
async def test_outputter_with_natural_language_output(mock_get_llm, mock_context, mock_state):
    """Test outputter with natural language description output."""
    # Mock the LLM response for natural language output (no structured output)
    mock_llm = AsyncMock()
    expected_json = '{"color": "green", "price": 20, "currency_symbol": "$", "website_url": "http://superwebsite.fr"}'
    mock_llm.ainvoke.return_value = Mock(content=expected_json)
    mock_get_llm.return_value = mock_llm

    config = OutputConfig(
        structured_output=None,
        output_description=(
            "A JSON object with a color, a price, a currency_symbol and a website_url key"
        ),
    )

    result = await outputter(ctx=mock_context, output_config=config, graph_output=mock_state)

    assert isinstance(result, dict)
    assert result.get("color") == "green"
    assert result.get("price") == 20
    assert result.get("currency_symbol") == "$"
    assert result.get("website_url") == "http://superwebsite.fr"
