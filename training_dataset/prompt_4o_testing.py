# PROMPT_PAPER = """
# # # You are a helpful assistant for automatic automatic query generation. 
# # # You'll read a paragraph, and then issue queries of different length and complexity in the Serbian language to a search engine in order to fact-check it. Also explain the queries.
# # # Finally, assign a score in a range from 1 to 5. The score description is provided below.
# # # The JSON object must contain the following:
# # # keys:
# # # {{"keywords" : a list of 5 relevant keyword strings.,
# # #   "short_query": a string, a context-related user search query. Limit your resposnse here to 2-4 words.,
# # #   "medium query: a string, a context-related user search. query Limit your resposnse here to 5-7 words.,
# # #   "long_query": a string, a context-related user search query. Limit your resposnse here to 8-10 words.,
# # #   "reasoining": An explanation of the reasoning behind the generated questions and answers for the given context.,
# # #   "scores": {{short_query": A  score from 1 to 5 based on the score description. It is the relatedness of short query and the given context,
# # #              "medium_query": A  score from 1 to 5 based on the score description. It is the relatedness of medium query and the given context,
# # #              "long_query": A  score from 1 to 5 based on the score description. It is the relatedness of long query and the given context.
# # #     }}
# # # }}

# # # Score Description:
# # # - A score represents the relatedness of the context to a query. You must output a score for each query, objectively and without bias.
# # # - Scores must be on a scale from 1 to 5, where:
# # #     - 1 = The answer cannot be found in the context.
# # #     - 2 = The answer is unclear from the context. The information in the context is scarce, making the answer difficult to determine.
# # #     - 3 = Inferring the answer from the context requires interpretation. The context provides some relevant information.
# # #     - 4 = The answer is understandable from the context. The context provides sufficient information to comprehend the answer.
# # #     - 5 = The answer is directly and explicitly provided in the context.

# # # Please adhere to the following guidelines:
# # # - All queries should be in the Serbian language.
# # # - The three queries should not pertain to the exact same informaton from the paragraph. Generate diverse queries where possible.
# # # - If a query begins with a question word, it must end with a questionmark (?).
# # # - All queries must adhire to the length ranges defined in the JSON output example.
# # # - Lengths should have a normal distrubtion.
# # # - All the queries reqire high-school level education to understand.

# # # Your output must always be a JSON object only.
# # # Let's think step by step.
# # # Be creative!
# # # Context:
# # # {context}
# """

# PROMPT_PAPER = """
# # # You are a helpful assistant for automatic query generation. 
# # # You'll read a paragraph, and then issue queries of different lengths and complexity in the Serbian language to a search engine in order to fact-check it. Also explain the queries.
# # # Finally, assign a score in a range from 1 to 5. The score description is provided below.
# # # The JSON object must contain the following:
# # # {{"keywords" : a list of 5 relevant keyword strings.,
# # #   "short_query": a string, a context-related user search query. Short query length MUST BE between 2-4 words.,
# # #   "medium query: a string, a context-related user search. query Meidum query length MUST BE between 5-7 words.,
# # #   "long_query": a string, a context-related user search query. Long query length MUST BE between 8-10 words.,
# # #   "reasoining": An explanation of the reasoning behind the generated questions and answers for the given context.,
# # #   "scores": {{short_query": A  score from 1 to 5 based on the score description. It is the relatedness of short query and the given context,
# # #              "medium_query": A  score from 1 to 5 based on the score description. It is the relatedness of medium query and the given context,
# # #              "long_query": A  score from 1 to 5 based on the score description. It is the relatedness of long query and the given context.
# # #     }}
# # # }}

# # # Score Description:
# # # - A score represents the relatedness of the context to a query. You must output a score for each query, objectively and without bias.
# # # - Scores must be on a scale from 1 to 5, where:
# # #     - 1 = The answer cannot be found in the context.
# # #     - 2 = The answer is unclear from the context. The information in the context is scarce, making the answer difficult to determine.
# # #     - 3 = Inferring the answer from the context requires interpretation. The context provides some relevant information.
# # #     - 4 = The answer is understandable from the context. The context provides sufficient information to comprehend the answer.
# # #     - 5 = The answer is directly and explicitly provided in the context.

# # # Please adhere to the following guidelines:
# # # - All queries, keywords, and reasoning should be in the Serbian language.
# # # - Queries can be in either question format or statement format. Aim for a mix of both formats across the three queries.
# # # - If a query begins with a question word, it must end with a questionmark (?).
# # # - The three queries must cover different aspects of the given context. They should not be rephrases of the same question or focus on the same piece of information.
# # # - All queries must adhere to the length ranges defined in the JSON output example:
# # #   * Short query: 3-4 words
# # #   * Medium query: 5-7 words
# # #   * Long query: 8-12 words

# # # - Double-check that each query falls within its specified word count range before finalizing your response.
# # # - The keywords should be directly related to the generated queries and the main topics of the context.
# # # - In the "reasoning" section, explain:
# # #   a. Why each query was chosen and how it relates to the context
# # #   b. How the queries differ from each other
# # #   c. Why the specific keywords were selected
# # #   d. Confirm that each query meets its required word count

# # # Your output must always be a JSON object only.
# # # Let's think step by step and closly follow the instructions.
# # # Context:
# # # {context}
# """

# PROMPT = """
# # You are a helpful assistant for automatic query generation. 

# # You'll read a paragraph and then generate queries of different lengths and complexities in the Serbian language to fact-check the information. Your task includes:

# # 1. Generating:
# #    - **Short Query:** A context-related user search query with 3-4 words.
# #    - **Medium Query:** A context-related user search query with 5-7 words.
# #    - **Long Query:** A context-related user search query with 8-12 words.

# # 2. Explaining:
# #    - **Reasoning:** Provide an explanation for why each query was generated, how it relates to the context, how the queries differ, and why the keywords were chosen.

# # 3. Scoring:
# #    - Assign a score from 1 to 5 for each query based on its relatedness to the context. Use the following scoring scale:
# #      - **1:** The answer cannot be found in the context.
# #      - **2:** The answer is unclear from the context; the information is scarce.
# #      - **3:** The answer requires interpretation; some relevant information is present.
# #      - **4:** The answer is understandable; sufficient information is provided.
# #      - **5:** The answer is directly and explicitly provided.

# # 4. Output Format:
# #    - Provide the result as a JSON object containing:

# #     {{"keywords" : ["keyword1", "keyword2", "keyword3", "keyword4", "keyword5"],
# #     "short_query": "Short query here",
# #     "medium query: "Medium query here",
# #     "long_query": "Long query here",
# #     "reasoining": "Explanation of reasoning"
# #     "scores": {{short_query": Short query score from the scoring scale (1-5).,
# #                 "medium_query": Medium query score from the scoring scale (1-5).,
# #                 "long_query": Long query score from the scoring scale (1-5).,
# #         }}
# #     }}

# # 5. Guidelines:
# #    - All content (queries, keywords, reasoning) must be in Serbian.
# #    - Queries can be in question or statement format. Use a mix of both.
# #    - Questions should end with a question mark.
# #    - The three queries should not pertain to the exact same informaton from the paragraph. Generate diverse queries where possible.
# #    - Verify that each query meets its specified word count range.
# #    - Keywords should directly relate to the main context topics.
# #    - Explain why each query was chosen, how it relates to the context, how the queries differ, and why the keywords were selected.

# # Context:
# # {context}
# # """

# PROMPT = """
# # You are a helpful assistant for automatic automatic query generation. 
# # You'll read a paragraph, and then issue queries in the Serbian language to a search engine in order to fact-check it. Also explain the queries.
# # Finally, assign a score in a range from 1 to 5. The score description is provided below.

# # ### Instructions ###
# # - Generate queries based only on the provided context, not the prior knowledge.
# # - Make sure that the queries are relevant to the context.
# # - Answers to the generated queries must have semantic meaning and be derived directly from the provided context.
# # - Score each output based on the scale below:

# # ### Score Description ###
# # - A score represents the relatedness of the context to a query. You must output a score for each query, objectively and without bias.
# # - Scores must be on a scale from 1 to 5, where:
# #     - 1 = The answer cannot be found in the context.
# #     - 2 = The answer is unclear from the context. The information in the context is scarce, making the answer difficult to determine.
# #     - 3 = Inferring the answer from the context requires interpretation. The context provides some relevant information.
# #     - 4 = The answer is understandable from the context. The context provides sufficient information to comprehend the answer.
# #     - 5 = The answer is directly and explicitly provided in the context.

# # ### Output format ###
# # - Output the data in a JSON format parsable by json.loads() in Python.

# # {{
# #  "keywords": ["Context keywords"],
# #  "short_query": "Limit your response to 2 to 4 words.",
# #  "medium_query": "Limit your response to 5 to 7 words.",
# #  "long_query": "Limit your response to 6 to 10 words.",
# #  "reasoning": "An explanation of the reasoning behind the generated questions and answers for the given context."
# # #  "scores": {{
# #     "short_query": A  score from 1 to 5 based on previos score description. It is the relatedness of short query and the given context,
# #     "medium_query": A  score from 1 to 5 based on previos score description. It is the relatedness of medium query and the given context,
# #     "long_query": A  score from 1 to 5 based on previos score description. It is the relatedness of long query and the given context.

# #  }}
# # }}

# # ### Context ###
# # {context}

# # ### Summary ###
# # To revise, you should: read the provided context, reason about its meaning, 
# # issue queries in the Serbian language to a search engine in order to fact-check it, explain the queries, 
# # assign a score to each query, and generate a parsable JSON output.
# # All the generated queries must adhire to the length limits predefined in the Output format.
# # Let's think step by step while doing the task.
# # """

# PROMPT_PAPER = """
# # You are a helpful assistant for automatic automatic query generation. 
# # You'll read a paragraph, and then issue queries of different length and complexity in the Serbian language to a search engine in order to fact-check it. Also explain the queries.
# # Finally, assign a score in a range from 1 to 5. The score description is provided below.
# # The JSON object must contain the following:
# # keys:
# # {{"keywords" : a list of 5 relevant keyword strings.,
# #   "short_query": a string, a context-related user search query. Limit your resposnse here to 2-4 words.,
# #   "medium query: a string, a context-related user search. query Limit your resposnse here to 5-7 words.,
# #   "long_query": a string, a context-related user search query. Limit your resposnse here to 8-10 words.,
# #   "reasoining": An explanation of the reasoning behind the generated questions and answers for the given context.,
# #   "scores": {{short_query": A  score from 1 to 5 based on the score description. It is the relatedness of short query and the given context,
# #              "medium_query": A  score from 1 to 5 based on the score description. It is the relatedness of medium query and the given context,
# #              "long_query": A  score from 1 to 5 based on the score description. It is the relatedness of long query and the given context.
# #     }}
# # }}

# # Score Description:
# # - A score represents the relatedness of the context to a query. You must output a score for each query, objectively and without bias.
# # - Scores must be on a scale from 1 to 5, where:
# #     - 1 = The answer cannot be found in the context.
# #     - 2 = The answer is unclear from the context. The information in the context is scarce, making the answer difficult to determine.
# #     - 3 = Inferring the answer from the context requires interpretation. The context provides some relevant information.
# #     - 4 = The answer is understandable from the context. The context provides sufficient information to comprehend the answer.
# #     - 5 = The answer is directly and explicitly provided in the context.

# # Please adhere to the following guidelines:
# # - All queries should be in the Serbian language.
# # - The three queries should not pertain to the exact same informaton from the paragraph. Generate diverse queries where possible.
# # - If a query begins with a question word, it must end with a questionmark (?).
# # - All queries must adhire to the length ranges defined in the JSON output example.
# # - Lengths should have a normal distrubtion.
# # - All the queries reqire high-school level education to understand.

# # Your output must always be a JSON object only.
# # Let's think step by step.
# # Be creative!
# # Context:
# # {context}
# # """

# PROMPT_PAPER = """
# You are a helpful assistant for automatic query generation. 
# You'll read a paragraph, and then issue queries of different lengths and complexity in the Serbian language to a search engine in order to fact-check it. Also explain the queries.
# Finally, assign a score in a range from 1 to 5. The score description is provided below.
# The JSON object must contain the following:
# {{"keywords" : a list of 5 relevant keyword strings.,
#   "short_query": a string, a context-related user search query. Short query length MUST BE between 2-4 words.,
#   "medium query: a string, a context-related user search. query Meidum query length MUST BE between 5-7 words.,
#   "long_query": a string, a context-related user search query. Long query length MUST BE between 8-10 words.,
#   "reasoining": An explanation of the reasoning behind the generated questions and answers for the given context.,
#   "scores": {{short_query": A  score from 1 to 5 based on the score description. It is the relatedness of short query and the given context,
#              "medium_query": A  score from 1 to 5 based on the score description. It is the relatedness of medium query and the given context,
#              "long_query": A  score from 1 to 5 based on the score description. It is the relatedness of long query and the given context.
#     }}
# }}

# Score Description:
# - A score represents the relatedness of the context to a query. You must output a score for each query, objectively and without bias.
# - Scores must be on a scale from 1 to 5, where:
#     - 1 = The answer cannot be found in the context.
#     - 2 = The answer is unclear from the context. The information in the context is scarce, making the answer difficult to determine.
#     - 3 = Inferring the answer from the context requires interpretation. The context provides some relevant information.
#     - 4 = The answer is understandable from the context. The context provides sufficient information to comprehend the answer.
#     - 5 = The answer is directly and explicitly provided in the context.

# Please adhere to the following guidelines:
# - All queries, keywords, and reasoning should be in the Serbian language.
# - Queries can be in either question format or statement format. Aim for a mix of both formats across the three queries.
# - If a query begins with a question word, it must end with a questionmark (?).
# - The three queries must cover different aspects of the given context. They should not be rephrases of the same question or focus on the same piece of information.
# - All queries must adhere to the length ranges defined in the JSON output example:
#   * Short query: 3-4 words
#   * Medium query: 5-7 words
#   * Long query: 8-12 words

# - Double-check that each query falls within its specified word count range before finalizing your response.
# - The keywords should be directly related to the generated queries and the main topics of the context.
# - In the "reasoning" section, explain:
#   a. Why each query was chosen and how it relates to the context
#   b. How the queries differ from each other
#   c. Why the specific keywords were selected
#   d. Confirm that each query meets its required word count

# Your output must always be a JSON object only.
# Let's think step by step and closly follow the instructions.
# Context:
# {context}
# """

# PROMPT = """
# You are a helpful assistant for automatic query generation. The main objective is to generate a query that is relevant to the provided context. 
# The generated queries and the respective contexts will be used to fine-tune an embedding model for Information Retrieval. 
# The goal is to create an expert, golden standard dataset to fine-tune an embedding model for Information retrieval.

# You'll read a paragraph and then generate queries of different lengths and complexities in the Serbian language to fact-check the information. Your task includes:

# 1. Generating:
#    - **Query:** A context-related user search query.

# 2. Explaining:
#    - **Reasoning:** Provide an explanation for why each query was generated, how it relates to the context, how the queries differ.

# 3. Output Format:
#    - Provide the result as a JSON object containing:

#     {{
#     "query": "Query.",
#     "reasoining": "Reasoning.",
#     }}

# 4. Guidelines:
#    - All content (queries, reasoning) must be in Serbian.
#    - Queries can be in question or statement format. Use a mix of both.
#    - Questions should end with a question mark.
#    - Generate diverse queries.
#    - Explain why each query was chosen, and how it relates to the context.

# Context:
# {context}

# To sum up, your task is to generate queries relevant to the provided contexts to obtain a golden standard dataset for Information Retrival task. 
# To accomplish this, you should carefully read the provided text, find relevant information, and generate a query that pertains to it. 
# A query can be either a question or a statement. You should reason about each query you generate.

# Let's think step by step and closely follow the provied instructions.
# """

PROMPT_A = """
You are a helpful assistant for automatic query generation. 

You'll read a paragraph and then generate queries of different lengths and complexities in the Serbian language to fact-check the information. Your task includes:

1. Generating:
   - **Short Query:** A context-related user search query with 3-4 words.
   - **Medium Query:** A context-related user search query with 5-7 words.
   - **Long Query:** A context-related user search query with 8-12 words.

2. Explaining:
   - **Reasoning:** Provide an explanation for why each query was generated, how it relates to the context, how the queries differ, and why the keywords were chosen.

3. Scoring:
   - Assign a score from 1 to 5 for each query based on its relatedness to the context. Use the following scale:
     - **1:** The answer cannot be found in the context.
     - **2:** The answer is unclear from the context; the information is scarce.
     - **3:** The answer requires interpretation; some relevant information is present.
     - **4:** The answer is understandable; sufficient information is provided.
     - **5:** The answer is directly and explicitly provided.

4. Output Format:
   - Provide the result as a JSON object containing:
 {{
  "keywords": ["Context keywords"],
  "short_query": "Limit your response to 2 to 4 words.",
  "medium_query": "Limit your response to 5 to 7 words.",
  "long_query": "Limit your response to 6 to 10 words.",
  "reasoning": "An explanation of the reasoning behind the generated questions and answers for the given context."
   "scores": {{
    "short_query": A  score from 1 to 5 based on previos score description. It is the relatedness of short query and the given context,
    "medium_query": A  score from 1 to 5 based on previos score description. It is the relatedness of medium query and the given context,
    "long_query": A  score from 1 to 5 based on previos score description. It is the relatedness of long query and the given context.

      }}
 }}

5. Guidelines:
   - All content (queries, keywords, reasoning) must be in Serbian.
   - Queries can be in question or statement format. Use a mix of both.
   - Questions should end with a question mark.
   - Ensure queries cover different aspects of the context and are not mere rephrases.
   - Verify that each query meets its specified word count range.
   - Keywords should directly relate to the queries and main context topics.
   - Explain why each query was chosen, how it relates to the context, how the queries differ, and why the keywords were selected.

Context:
{context}
"""