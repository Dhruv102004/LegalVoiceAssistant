import os
from google import genai
from google.genai import types

API_KEY = os.environ.get("GOOGLE_API_KEY", "")
FILE_STORE = os.environ.get("FILE_SEARCH_STORE_NAME", "")

client = genai.Client(api_key=API_KEY)

def run_search(query):
    custom_prefix = """
You must follow these rules for EVERY answer:
1. All responses must be contextualized for blind individuals in India.
2. Information must be India-specific whenever relevant.
3. Responses must be short, precise, factual, and based on real data.
"""
    final_query = custom_prefix + "\nQuery: " + query
    return client.models.generate_content(
        model="gemini-2.5-flash",
        contents=final_query,
        config=types.GenerateContentConfig(
            tools=[types.Tool(google_search=types.GoogleSearch())]
        )
    )

def query_with_fallback(query):
    file_response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=query+"for blind individuals in india",
        config=types.GenerateContentConfig(
            tools=[
                types.Tool(
                    file_search=types.FileSearch(
                        file_search_store_names=[FILE_STORE]
                    )
                )
            ]
        )
    )

    metadata = getattr(file_response.candidates[0], "grounding_metadata", None)
    has_chunks = metadata and getattr(metadata, "grounding_chunks", None)
    if not has_chunks:
        return run_search(query)

    relevance_prompt = f"""
Question: {query}

Response from documents:
{file_response.text}

Answer only "yes" or "no".
Say "yes" only if:
- a relevant, to-the point response to the query can be generated from the response and
- if this information is practically useful to the user

Otherwise say "no"!
"""
    relevance_check = client.models.generate_content(model="gemini-2.5-flash", contents=relevance_prompt)
    decision = relevance_check.text.strip().lower()
    if decision.startswith("no"):
        return run_search(query)
    return file_response
