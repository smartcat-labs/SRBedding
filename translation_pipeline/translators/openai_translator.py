import asyncio

import aiohttp
import structlog
import tiktoken
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_random_exponential

from translators.base import Translator

logger = structlog.get_logger()


class OpenAITranslator(Translator):
    def __init__(self, api_key: str, default_model="gpt-4o") -> None:
        self.client = OpenAI(api_key=api_key)

        self.model_max_tokens = {
            "gpt-4o": 128000,
            "gpt-4": 8192,
            "gpt-4-32k": 32768,
            "gpt-4-turbo": 128000,
            "gpt-3.5-turbo": 4096,
            "gpt-3.5-turbo-16k": 16384,
        }

        # Ratio of tokens in English vs target language
        self.token_ratio_to_language = {
            "serbian": 1.6,  # for Serbian language there will be 60% more tokens than in English!
        }

        self.default_model = default_model
        self.system_prompt = "Translate this text to \n"

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_random_exponential(multiplier=1, min=4, max=10),
    )
    async def _translate(self, session, text, language="serbian") -> str:
        prompt = self.system_message(text, language)
        payload = {
            "model": self.default_model,
            "prompt": prompt,
        }

        logger.info("Sending translation request", text=text)

        async with session.post(
            "https://api.openai.com/v1/completions",
            json=payload,
            headers={"Authorization": f"Bearer {self.api_key}"},
        ) as response:

            response_json = await response.json()
            translated_text = response_json["choices"][0]["text"].strip()

            logger.info(
                "Received translation", original=text, translation=translated_text
            )

            return translated_text

    async def translate_all(self, texts: list, language="serbian") -> list:
        """
        Translate a list of texts to a target language.
        """

        async with aiohttp.ClientSession() as session:
            tasks = [self._translate(session, text) for text in texts]
            logger.info("Translating batch of texts", num_texts=len(texts))
            translations = await asyncio.gather(*tasks)
            logger.info(
                "Batch translation complete", num_translations=len(translations)
            )
            return translations

    def system_message(self, language: str) -> str:
        """
        Generate a system message for the OpenAI API.
        """

        return f"{self.system_prompt} {language} language: \n"

    def estimate_tokens(
        self, text: str, model_used: str = None, target_language: str = "serbian"
    ) -> int:
        """
        Estimate the number of tokens required to translate a given text to a target language.

        Args:
        - text (str): Text to translate in `English`.
        - model_used (str): Model to use for translation.
        - target_language (str): Target language for translation.

        Returns:
        - int: Number of tokens required for translation. Sum of input and output tokens.
        """

        encoding = tiktoken.encoding_for_model(model_used or self.default_model)
        input_tokens = len(encoding.encode(text)) + len(
            encoding.encode(self.system_message(target_language))
        )

        output_tokens = int(
            input_tokens * self.token_ratio_to_language[target_language]
        )

        return input_tokens + output_tokens

    def estimate_cost(
        self, text: str, model_used: str = None, target_language: str = "serbian"
    ) -> float:
        """
        Estimate the cost of translating a given text to a target language.
        """

        raise NotImplementedError

    def max_tokens(self, model="gpt-4o") -> int:
        """
        Return the maximum number of tokens allowed by the OpenAI API.
        """

        return self.model_max_tokens[model]
