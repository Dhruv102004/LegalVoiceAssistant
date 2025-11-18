import os
from google import genai
from google.genai import types

API_KEY = os.environ.get("GOOGLE_API_KEY", "")
API_KEY_2 = os.environ.get("GOOGLE_API_KEY_2", "")
FILE_STORE = os.environ.get("FILE_SEARCH_STORE_NAME", "")

client = genai.Client(api_key=API_KEY)
_client = genai.Client(api_key=API_KEY_2)

def run_search(query):
    custom_prefix = """
You must follow these rules for EVERY answer:
1. All responses must be contextualized for blind individuals in India.
2. Information must be India-specific whenever relevant.
3. Responses must be short, precise, factual, and based on real data.
4. Responses must be less than 200 words STRICTLY, DO NOT GENERATE ANSWERS GREATER THAN 200 WORDS.
""" 
    final_query = custom_prefix + "\nQuery: " + query
    return _client.models.generate_content(
        model="gemini-2.5-flash",
        contents=final_query,
        config=types.GenerateContentConfig(
            tools=[types.Tool(google_search=types.GoogleSearch())]
        )
    )

def query_with_fallback(query):
    file_response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=query + " for blind individuals in india keep output less than 200 words STRICTLY",
        config=types.GenerateContentConfig(
            max_output_tokens=300,
            tools=[
                types.Tool(
                    file_search=types.FileSearch(
                        file_search_store_names=[FILE_STORE]
                    )
                )
            ]
        )
    )

    # Check grounding
    metadata = getattr(file_response.candidates[0], "grounding_metadata", None)
    has_chunks = metadata and getattr(metadata, "grounding_chunks", None)
    if not has_chunks:
        return run_search(query)

    # NEW: check if phrase occurs anywhere in the text
    if "provided" in file_response.text:
        return run_search(query)

    return file_response

