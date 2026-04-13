"""
测试 SemanticMatcher 的意图-工具语义匹配
"""
import pytest
from src.semantic_matcher import SemanticMatcher
from src.schemas.models import ToolMetadata


@pytest.mark.slow
@pytest.mark.skip(reason="Requires downloading BGE model; run manually with --runslow")
def test_semantic_match_weather():
    matcher = SemanticMatcher()
    tools = [
        ToolMetadata(name="read_file", description="read text file content"),
        ToolMetadata(name="weather_get", description="get weather information for a city"),
        ToolMetadata(name="calculate", description="perform mathematical calculations"),
    ]
    matcher.index(tools)
    results = matcher.match("帮我查一下今天北京的天气", top_k=2)
    assert len(results) > 0
    assert results[0][0].name == "weather_get"


def test_semantic_match_empty_index():
    matcher = SemanticMatcher()
    matcher.index([])
    results = matcher.match("any query")
    assert results == []
