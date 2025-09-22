import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint, HuggingFaceEmbeddings
from dotenv import load_dotenv
from pinecone import Pinecone as PineconeClient, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain.schema import Document
from langchain_core.tools import StructuredTool, Tool
from langchain_core.prompts import PromptTemplate
import json
import re
from typing import Any
from langchain.schema import AIMessage
load_dotenv()

df = pd.read_csv('fifa_players.csv')

df.drop(columns=['national_team','national_team_position','national_jersey_number'], inplace=True)

df.drop(columns=['national_rating'], inplace=True)

medians = df.median(numeric_only=True)


df = df.fillna(medians)

embeddings = HuggingFaceEmbeddings(model='sentence-transformers/all-MiniLM-L6-v2')

api_key=os.getenv("PINECONE_API_KEY")

index_name = 'ttyd-1'
vector_store = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

repo_id="openai/gpt-oss-20b"

llm = HuggingFaceEndpoint(
    repo_id=repo_id,
    task='text-generation'
)

model = ChatHuggingFace(llm=llm)

retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k':4})

def get_relevant_docs(query):
  docs = retriever.get_relevant_documents(query)
  context = "\n\n".join(doc.page_content for doc in docs)
  return context

def prompt_one(docs, query):
  prompt = PromptTemplate(
      template = '''
    You are an assistant that writes a SINGLE-LINE pandas command to answer the query.
    - Assume the DataFrame is called df.
    - Use ONLY pandas/numpy.
    - Do not add explanations, comments, Markdown, or extra text.
    - Output exactly one Python expression that evaluates to a DataFrame.

    Context (columns): {context}
    User query: {query}

    Example output:
    df[['name','overall_rating']].sort_values('overall_rating', ascending=False).head(5)
    ''',
        input_variables=['context', 'query']
  )
  return prompt.format(context=docs, query=query)

def generate_code(docs,query):
  template = prompt_one(docs, query)
  result = model.invoke(template)
  return result.content

def execute_pandas(code: str):
    try:
        result = eval(code, {"df": df, "pd": pd, "np": np})
        if isinstance(result, pd.Series):
            result = result.to_frame()
        return {
            "columns": list(result.columns),
            "rows": result.to_dict(orient="records")
        }
    except Exception as e:
        return {"error": str(e)}
    
    
def summarise(docs, query):
  json_context = json.dumps(docs, indent=2)
  prompt = PromptTemplate(
    input_variables=["context", "query"],
    template="""
    You are an assistant that analyzes tabular data.
    Your goal is to summarise the data to answer a question.

    The following is the data (in JSON):
    {context}

    User question:
    {query}

    Answer the question using only the data provided.
    """
  )

  final_prompt = prompt.format(context=json_context, query=query)
  result = model.invoke(final_prompt)
  return result.content

def _unwrap_llm_output(raw_out: Any) -> str:
    """
    Convert common LangChain/HuggingFace responses into a string safely.
    Handles:
      - AIMessage (has .content)
      - list/tuple of AIMessage
      - dict-like with 'content' or nested fields
      - plain str
    """
    # AIMessage (langchain)
    try:
        if hasattr(raw_out, "content"):
            return str(raw_out.content)
    except Exception:
        pass

    # list/tuple of messages
    if isinstance(raw_out, (list, tuple)) and len(raw_out) > 0:
        first = raw_out[0]
        if hasattr(first, "content"):
            return str(first.content)
        return str(first)

    # dict-like
    if isinstance(raw_out, dict):
        # try common spots
        for key in ("content", "text", "message", "result"):
            if key in raw_out and isinstance(raw_out[key], str):
                return raw_out[key]
        # nested generational style (LangChain LLMResult-like)
        if "generations" in raw_out and isinstance(raw_out["generations"], list):
            # pick first generation, first text field
            try:
                gen = raw_out["generations"][0][0]
                if isinstance(gen, dict) and "text" in gen:
                    return gen["text"]
            except Exception:
                pass
        # fallback
        return json.dumps(raw_out)

    # plain string fallback
    return str(raw_out)


def json_to_vegalite_spec(json_result, user_query, llm):
    """
    Minimal wrapper that asks your HuggingFace LangChain LLM to return a Vega-Lite spec.
    `llm` may be a HuggingFaceEndpoint or any LangChain LLM that supports .invoke(...) or .generate(...) / .__call__.
    """
    # serialize json_result for prompt
    if not isinstance(json_result, str):
        json_payload = json.dumps(json_result, default=str)
    else:
        json_payload = json_result

    prompt = f"""
    You are a data visualization assistant.
    Decide the best plot to use to answer the user query.
    Convert the given JSON data and user query into a valid Vega-Lite v5 specification according to the plot you have chosen.

    JSON data:
    {json_payload}

    User query: "{user_query}"

    Requirements:
    - Return ONLY a single JSON object.
    - Use "$schema": "https://vega.github.io/schema/vega-lite/v5.json".
    - Put the provided data under "data": {{ "values": [...] }}.
    - Infer correct field types: temporal, quantitative, nominal.
    - If you cannot create a chart, return a table spec like {{ "table": {{ "values": [...] }} }}.
    """

    # call the LLM (support a few common call styles)
    raw = None
    # prefer direct invoke if available
    if hasattr(llm, "invoke"):
        raw = llm.invoke(prompt)
    else:
        # try calling / __call__ / generate patterns
        try:
            raw = llm(prompt)
        except Exception:
            try:
                out = llm.generate([prompt])
                # langchain LLM.generate returns an LLMResult; try to extract text
                raw = out
            except Exception as e:
                raise RuntimeError("Could not call provided llm object: " + str(e))

    text = _unwrap_llm_output(raw).strip()

    # try direct JSON parse; if fails try to extract JSON substring
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"(\{.*\})", text, flags=re.S)
        if m:
            try:
                return json.loads(m.group(1))
            except Exception as e:
                raise ValueError(f"Failed to parse JSON from LLM output: {e}\nRaw text:\n{text}")
        raise ValueError(f"LLM did not return parseable JSON.\nRaw text:\n{text}")
    
def test_pipeline(query: str):
    # 1. Retrieve relevant columns
    context = get_relevant_docs(query)

    # 2. Build prompt (not really needed since generate_code calls prompt_one internally)
    # prompt = prompt_one(context, query)

    # 3. LLM generates pandas code
    code = generate_code(context, query)

    # 4. Execute code on df → JSON
    result_json = execute_pandas(code)

    spec = json_to_vegalite_spec(result_json, query, model)

    return spec