import sys
from unittest.mock import AsyncMock, Mock, patch

import pytest
from pydantic import BaseModel

# 这些 stub 只为让下面那几行 import 跑得起来。
# **import 完必须原样还回去**：留在 sys.modules 里会永久污染同一个 pytest 进程，
# 让后续收集的测试模块看到假的父包。实测后果是
# tests/mobile_use/test_deepseek_provider.py 收集期直接报
# "'langchain_openai' is not a package" —— 单跑它是通过的。
_STUBBED = (
    "langgraph.prebuilt.chat_agent_executor",
    "minitap.mobile_use.graph.state",
    "langchain_google_vertexai",
    "langchain_google_genai",
    "langchain_openai",
    "langchain_cerebras",
    "langchain_azure_ai",
    "langchain_azure_ai.chat_models",
)
_saved_modules = {name: sys.modules.get(name) for name in _STUBBED}
for _name in _STUBBED:
    sys.modules[_name] = Mock()

from minitap.mobile_use.agents.outputter.outputter import outputter  # noqa: E402
from minitap.mobile_use.config import LLM, OutputConfig  # noqa: E402
from minitap.mobile_use.context import MobileUseContext  # noqa: E402
from minitap.mobile_use.utils.logger import get_logger  # noqa: E402

# 先删子包再删父包，避免留下指向已删除父包的孤儿条目。
for _name in reversed(_STUBBED):
    _original = _saved_modules[_name]
    if _original is not None:
        sys.modules[_name] = _original
    else:
        sys.modules.pop(_name, None)

logger = get_logger(__name__)


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
    def __init__(self, messages, initial_goal, agents_thoughts):
        self.messages = messages
        self.initial_goal = initial_goal
        self.agents_thoughts = agents_thoughts


mocked_state = DummyState(
    messages=[],
    initial_goal="Find a green product on my website",
    agents_thoughts=[
        "Going on http://superwebsite.fr",
        "Searching for products",
        "Filtering by color",
        "Color 'green' found for a 20 dollars product",
    ],
)


@pytest.fixture
def mock_context():
    """Create a properly mocked context with all required fields."""
    ctx = Mock(spec=MobileUseContext)
    ctx.llm_config = {
        "executor": LLM(provider="openai", model="gpt-5-nano"),
        "cortex": LLM(provider="openai", model="gpt-5-nano"),
        "planner": LLM(provider="openai", model="gpt-5-nano"),
        "orchestrator": LLM(provider="openai", model="gpt-5-nano"),
    }
    ctx.device = Mock()
    return ctx


@pytest.fixture
def mock_state():
    """Create a mock state with test data."""
    return DummyState(
        messages=[],
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
