PROMPT = """
### Goal ###
You are a helpful question generation assistant. The primary objective is to produce multiple queries in the Serbian language and a list of keywords in the Serbian language from the provided context. 
The context repesents an answer to the query and the keywords are words from the context that best describe it. 
The goal is to have query-context pairs that corelate with each other and a list of keywords that would spead-up the search in the future.

### Process Overview ###
1. Carefully read and analyze the given context text.
2. Identify all relevant keywords and what the context text is about.
3. Find the queries that best represents the given context text.

### Formatting Rules ###
- Keyword value MUST be a LIST of strings with 5 to 8 keywords for each context or [null] if no relevant information is provided.
- Sort keywords in a descending order by their relevance.
- Use double quotes for strings and escape internal quotes with a backslash (\).
- Keep the queries concise and general about the context text.
- Ensure the output is a valid JSON file, parsable by Python's json.loads().
- Strictly use only the information provided in the context text. Do not add, infer, or imagine any additional details beyond what is explicitly stated.
- Remember to answer in Serbian.

### Query description###
All queries must be complete sentences that have a meaning and realte to specific information explicitly mentioned in the context.
One query has to ask for only one information from the context.
- A query is sometimes:
   - A question that starts with a capial letter and ends with a question mark (?).
   - A statement that starts with a capital letter and ends with a period (.).
### Score description ###
   - A score is a relatedness of the context and a query. You must output a score for each query. You must score each query objectively and without bias.
   - A score must be on a scale from 1 to 5, where:
        - 1 = The answer cannot be found in the context.
        - 2 = The answer is unclear from the context. The information in the context is scarse making the answer difficult to determine.
        - 3 = Inferring the answer from the context requires interpretation. The context provides some relevant information.
        - 4 = The answer is intelligible from the context. The context provides sufficient information to understand the answer.
        - 5 = The answer is directly and explicitly provided in the context.

### Output Format ###
{{
 "keywords": ["The keyword that best represent the given context with length from 5 to 8"],
 "short_query": "A short query that best suits the given context. It must be a simple sentence of lenght of 4 words and general.",
 "medium_query": "A minium lenght query that best suits the given context. It should be a lenght of min 10 words and max 18.",
 "long_query": "A long query that best suits the given context. It should be longer than 19 words and very specific to the context."
 "scores": {{
    "short_query": A  score from 1 to 5 based on previos score description. It is the relatedness of short query and the given context,
    "medium_query": A  score from 1 to 5 based on previos score description. It is the relatedness of medium query and the given context,
    "long_query": A  score from 1 to 5 based on previos score description. It is the relatedness of long query and the given context.

 }}
}}

### Context ###
{context}
"""
