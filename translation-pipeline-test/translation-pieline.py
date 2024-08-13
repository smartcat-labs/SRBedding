import transformers
import sentence_transformers
import os
import openai
import sys
from openai import OpenAI
from datasets import load_dataset
from pprint import pprint
import json

def make_jobs(model, prompt, filename, dataset):
    jobs = [
                {
                    "model": model,
                    "response_format": {'type': "json_object"},
                    "temperature": 0,
                    "metadata": {"query_id": sample['query_id']},
                    "messages": [
                        {
                            "role": "system",
                            "content": prompt
                        },
                           {
                             "role": "user",
                             "content": json.dumps(sample)
                        }
                    ],
                    
                }
                for sample in dataset
            ]
    with open(filename, "w") as f:
        for job in jobs:
            json_string = json.dumps(job)
            f.write(json_string + "\n")
    

model = 'gpt-3.5-turbo-0125'
make_jobs(model, SYSTEM_PROMPT, 'test', final_data)



def load_data(dataset_name="microsoft/ms_marco"):
    data = load_dataset(dataset_name, "v1.1")
    data_test_split = data['test']
    ms_marco = data_test_split.select_columns(['passages', 'query', 'query_id'])

    final_data = []
    for i in range(3):
        final_data.append(
            {
                'query_id' : ms_marco['query_id'][i],   
                'query' : ms_marco['query'][i],  
                'passage_text': ms_marco['passages'][i]['passage_text']
            }
        )

    return final_data


SYSTEM_PROMPT = """
***TRANSLATION FROM ENGLISH TO SERBIAN***

**GOALS**

You are a professional translator fluent in English and Serbian. 
Your primary goal is to produce a high-quality, natural-sounding translation from English to Serbian. 
You are translating texts and questions pertaining to the texts. The translation is intended for dataset creation. Look at the example below:

***TRANSLATION EXAMPLE***
***ENGLISH***
query: 'What is a unicorn?'
passage_text: 'The unicorn is a legendary creature that has been described since antiquity as a beast with a single large, pointed, spiraling horn projecting from its forehead.'

***SERBIAN TRANSLATION***
query: 'Šta je jednorog?'
passage_text: 'Jednorog je mitsko stvorenje koje se od davnina opisuje kao zver sa jednim velikim, šiljastim, spiralnim rogom koji mu viri iz čela.'
***END OF TRANSLATION EXAMPLE***

To translate, follow the steps below:
   **TRANSLATION INSTRUCTIONS**
   1. Read and understand the sentence in English.
   2. When you understand the English sentence, start to translate.
   3. Pay close attention to both left and right context when you are making translation decisions.  4. Convey the original context, tone and meaning in the Serbian translation.
   4. Avoid literal translations and ensure the output reads naturally in Serbian.
   5. The translation must be contextually accurate, fluent, and adhere to the grammatical rules and lexicon of the Serbian language.
   6. The declination of nouns, adjectives, and pronouns must be correct.
   7. Make sure to proofread the translated text in Serbian and revise any mistakes. If no revisions are needed, provide the translations as they are.

   **FORMATTING INSTRUCTIONS**
   1. Strings should be enclosed within single quotation marks ('').
   2. Use double quotes for strings and escape internal quotes with a backslashes (\).
 
   **OUTPUT FORMATTING**
   1. The output must be a valid JSON file.
   2. The output must be parsable by Python's 'json.loads()' function.
   3. Ensure consistent JSON formatting as illustrated in the example below:

      **EXAMPLE**
      {'translations' : [{'query' : 'This is a query',
                      ''passage_text ' : {'This is one passage. With another sentence.',
                                          'This is yet another passage. With yet another sentence.'}]
      }
       **END OF EXAMPLE**

   - Strictly follow the structure provided in the example when generating the output.   
   - Make sure to translate text under both "query" and "passage_text" keys.

"""

if __name__ == "__main__":
    OPENAI_KEY = os.environ['OPENAI_API_KEY']
    client = OpenAI(api_key=OPENAI_KEY)
