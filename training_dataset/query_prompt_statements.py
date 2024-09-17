PROMPT = """
### Goal ###
Act as a user searching for information from text using natural language queries. 
Imagine that the contexts you get are the documents you would like the search engine to retrieve based on your user query.
Your primary objective is to produce multiple queries in the Serbian language from the provided context.
The context represents an answer to a the user query. The goal is to get a dataset with user queries and contexts that the user queries pertain to.

### Process Overview ###
1. Carefully read and analyze the given context text.
2. Identify all relevant keywords that best describe the content of the context text.
3. Based on this information, act as a user that would try to search for that information (e.g. on Google) and generate the user query that pertains to it.
4. One user query must pertain to only one relevant information in the given context.
5. Make sure that the user queries are relevant and answerable from the context.
6. Avod user queries where answers are not semantic words.
7. Give the reasoinign steps in the Serbian language for each generated question.

### Instructions ###
- Use double quotes for strings and escape internal quotes with a backslash (\).
- The answers to the generated user queries must be obvions from the provided contexts.
- Strictly use only the information provided in the context text. Do not add, infer, or imagine any details beyond what is explicitly stated.
- All generated text must be in the Serbian language.
- You must provide reasoning in Serbian for each generated question.
- Ensure the output is a valid JSON format, parsable by Python's `json.loads()`.

### User Query Description ###
- The user queries should be information-oriented searches, i.e. user queries that target obtaining specific information from the dataset.
- Short, medium, and long queries should vary in length between categories and within the categories.

### Score Description ###
- A score represents the relatedness of the context to a query. You must output a score for each query, objectively and without bias.
- Scores must be on a scale from 1 to 5, where:
    - 1 = The answer cannot be found in the context.
    - 2 = The answer is unclear from the context. The information in the context is scarce, making the answer difficult to determine.
    - 3 = Inferring the answer from the context requires interpretation. The context provides some relevant information.
    - 4 = The answer is understandable from the context. The context provides sufficient information to comprehend the answer.
    - 5 = The answer is directly and explicitly provided in the context.

### Output Format ###
{{
 "keywords": ["The keyword that best represent the given context with length from 5 to 8"],
 "short_query": "A short user query that best suits the given context.",
 "medium_query": "A medium lenght query that best suits the given context.",
 "long_query": "A long query that best suits the given context.",
 "reasoning": "An explanation of the reasoning behind the generated questions for the given context."
 "scores": {{
    "short_query": A  score from 1 to 5 based on previos score description. It is the relatedness of short query and the given context,
    "medium_query": A  score from 1 to 5 based on previos score description. It is the relatedness of medium query and the given context,
    "long_query": A  score from 1 to 5 based on previos score description. It is the relatedness of long query and the given context.

 }}
}}

### Context ###
{context}

  """


# - 1 = A human could not find the answer from the context.
# - 2 = A human would have diffictuly deriving the answer from the context.
# - 3 = A human could interpret the answer from the context.
# - 4 = A human could easily derive the answer from the context.
# - 5 = A human could immediatly give the answer from the context.


