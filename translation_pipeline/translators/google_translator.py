import cyrtranslit
from google.cloud import translate_v2 as translate

# TODO: Finish the GoogleTranslator class


class GoogleTranslator:
    def __init__(self, api_key):
        self.api_key = api_key
        self.client = translate.Client()

    def translate(self, text: str, target_language: str = "serbian") -> str:
        """
        Translate text to a given language using Google Translate API.
        """

        translated_text = self.client.translate(
            text, target_language=target_language
        ).text

        return cyrtranslit.to_latin(translated_text, "sr")


# NOTE: Since GTranslate retrurns cyrillic text, we need to convert it to latinic
# `cyrtranslit` library is added for these purposes
