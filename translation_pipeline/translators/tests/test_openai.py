import pytest
from translators.openai_translator import OpenAITranslator


@pytest.fixture
def translator():
    return OpenAITranslator(api_key="your_api_key")


def test_system_message(translator):
    language = "german"
    system_message = translator.system_message(language)
    assert isinstance(system_message, str)
    assert system_message != ""


def test_estimate_tokens(translator):
    text = "This is a test"
    target_language = "serbian"
    estimated_tokens = translator.estimate_tokens(text, None, target_language)
    assert isinstance(estimated_tokens, int)
    assert estimated_tokens > 0


def test_max_tokens(translator):
    model = "gpt-4o"
    max_tokens = translator.max_tokens(model)
    assert isinstance(max_tokens, int)
    assert max_tokens > 4096
