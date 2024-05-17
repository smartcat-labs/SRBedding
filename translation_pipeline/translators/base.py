from abc import ABC, abstractmethod


class Translator(ABC):
    @abstractmethod
    def _translate(self, text: str) -> str:
        pass

    @abstractmethod
    def translate_all(self, texts: list) -> list:
        pass

    @abstractmethod
    def max_tokens(self) -> int:
        pass


class TranslatorFactory:
    @staticmethod
    def create_translator(translator_type: str) -> Translator:
        if translator_type == "openai":
            from translators.openai_translator import OpenAITranslator

            return OpenAITranslator()
        elif translator_type == "google":
            from translators.google_translator import GoogleTranslator

            return GoogleTranslator()
        else:
            raise ValueError("Invalid translator type!")
