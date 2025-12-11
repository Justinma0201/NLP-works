import os
import json
import bs4
import nltk
import torch
import pickle
import numpy as np

# from pyserini.index import IndexWriter
# from pyserini.search import SimpleSearcher
from numpy.linalg import norm
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize

from langchain_community.llms import Ollama
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.vectorstores import Chroma
from sentence_transformers import SentenceTransformer
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.embeddings import JinaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter, TokenTextSplitter
from langchain.docstore.document import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import WebBaseLoader
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

from tqdm import tqdm

from huggingface_hub import login

hf_token = " "
login(token=hf_token, add_to_git_credential=True)

MODEL = "llama3.2:1b" # https://ollama.com/library/llama3.2:3b
EMBED_MODEL = "jinaai/jina-embeddings-v2-base-en"


llm = Ollama(model=MODEL,
             temperature=0)

response = llm.invoke("What is the capital of Taiwan?")
print(response)


with open('cat-facts.txt', 'r', encoding='utf-8') as f:
  refs = [line.strip() for line in f if line.strip()]


from langchain_core.documents import Document
docs = [Document(page_content=doc, metadata={"id": i}) for i, doc in enumerate(refs)]

model_kwargs = {'trust_remote_code': True}
encode_kwargs = {'normalize_embeddings': False}
embeddings_model = HuggingFaceEmbeddings(
    model_name=EMBED_MODEL,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)


vector_store = Chroma.from_documents(
    documents=docs,
    embedding=embeddings_model

)
retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}
)


from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

model = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")

compressor = CrossEncoderReranker(model=model, top_n=5) #ai

rerank_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=retriever
)

system_prompt = (
"You are a strict data extraction assistant.\n"
"Task: Answer the question by copying the exact phrase from the provided context.\n"
"\n"
"Rules:\n"
"1. DIRECT QUOTE: Copy the answer exactly as written in the text. Do not rephrase.\n"
"2. INCLUDE QUALIFIERS: If the text says 'about', 'approx', 'up to', or units like 'mph', KEEP THEM.\n"
"3. NO MATH: If text says 'Two thirds', do NOT write '16 hours'. Copy 'Two thirds'.\n"
"4. FOCUS ON CATS: Ignore information about dogs or humans.\n"
"5. LISTS: If listing items (like colors), do NOT use commas. Just use spaces (e.g. 'Blue green red').\n"
"6. LENGTH: Keep it under 3 words.\n"
"\n"
"Context:\n{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)



question_answer_chain = create_stuff_documents_chain(llm, prompt)

chain = create_retrieval_chain(rerank_retriever, question_answer_chain)

from operator import itemgetter
from langchain_core.runnables import RunnablePassthrough, RunnableParallel

def reverse_docs_order(docs):
    return docs[::-1]

retrieval_step = RunnableParallel(
    context=itemgetter("input") | retriever | reverse_docs_order,
    input=itemgetter("input")
)

reverse_chain = retrieval_step.assign(
    answer=prompt | llm
)

queries = [
# Questions queries
]
answers = [
# Corresponding answers
]
if os.path.exists('questions_answers.txt'):
    with open('questions_answers.txt', 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]

        for i in range(0, len(lines), 2): #ai
            if i + 1 < len(lines):
                queries.append(lines[i])
                answers.append(lines[i+1])

for q, a in zip(queries[:2], answers[:2]):
    print(f"Q: {q}")
    print(f"A: {a}")
    print("---")

import string
import re
from tqdm import tqdm
from langchain_core.documents import Document
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_core.runnables import RunnablePassthrough

def normalize_answer(s):
    """Lower text and remove punctuation, articles and ALL whitespace."""

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def remove_all_spaces(text):
        return "".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return remove_all_spaces(remove_articles(remove_punc(lower(s))))

import string
import re
import json
from tqdm import tqdm

final_results = []
final_score_count = 0
recall_1_count = 0
recall_5_count = 0

for i, query in tqdm(enumerate(queries), total=len(queries)):
    response = chain.invoke({"input": query})
    pred = response["answer"]
    retrieved_docs = response["context"]
    gt_answer = answers[i]

    is_correct = False

    norm_pred = normalize_answer(pred)
    norm_gt = normalize_answer(gt_answer)

    if norm_gt in norm_pred:
        is_correct = True

    if is_correct:
        final_score_count += 1

    target_doc_id = i
    retrieved_ids = [doc.metadata.get("id") for doc in retrieved_docs]

    if len(retrieved_ids) > 0 and retrieved_ids[0] == target_doc_id:
        recall_1_count += 1
    if target_doc_id in retrieved_ids:
        recall_5_count += 1

    final_results.append({
        "Query": query,
        "Ground_Truth": gt_answer,
        "Prediction": pred,
        "Is_Correct": is_correct
    })

print("\n===== 錯誤分析 (前 5 筆) =====")
error_count = 0
for res in final_results:
    if not res['Is_Correct']:
        print(f"Q: {res['Query']}")
        print(f"GT: {res['Ground_Truth']}")
        print(f"Pred: {res['Prediction']}")
        print("-" * 30)
        error_count += 1
        if error_count >= 5:
            break

total_q = len(queries)
accuracy_score = final_score_count / total_q
recall_1_score = recall_1_count / total_q
recall_5_score = recall_5_count / total_q

print(f"\n===== Evaluation Results =====")
print(f"Total Questions: {total_q}")
print(f"Accuracy (Combined Logic): {accuracy_score:.2%}")
print(f"Recall@1: {recall_1_score:.2%}")
print(f"Recall@5: {recall_5_score:.2%}")

json_filename = "NLP_HW4_NTHU_114064558.json"

with open(json_filename, "w", encoding="utf-8") as f:
    json.dump(final_results, f, ensure_ascii=False, indent=4)

import json
from tqdm import tqdm

final_results = []
correct_count = 0
recall_1_count = 0
recall_5_count = 0

for i, query in tqdm(enumerate(queries), total=len(queries)):
    response = reverse_chain.invoke({"input": query})
    pred = response["answer"]
    retrieved_docs = response["context"]
    gt_answer = answers[i]

    norm_pred = normalize_answer(pred)
    norm_gt = normalize_answer(gt_answer)

    if norm_gt in norm_pred:
        is_correct = True
    else:
        is_correct = False

    if is_correct:
        correct_count += 1

    target_doc_id = i
    retrieved_ids = [doc.metadata.get("id") for doc in retrieved_docs]

    if len(retrieved_ids) > 0 and retrieved_ids[0] == target_doc_id:
        recall_1_count += 1
    if target_doc_id in retrieved_ids:
        recall_5_count += 1

    final_results.append({
        "Query": query,
        "Ground_Truth": gt_answer,
        "Prediction": pred,
        "Is_Correct": is_correct
    })

print("\n===== 錯誤分析 (前 5 筆) =====")
error_count = 0
for res in final_results:

    if not res['Is_Correct']:
        print(f"Q: {res['Query']}")
        print(f"GT: {res['Ground_Truth']}")
        print(f"Pred: {res['Prediction']}")
        print("-" * 30)
        error_count += 1
        if error_count >= 5:
            break

total_q = len(queries)
accuracy_score = correct_count / total_q
recall_1_score = recall_1_count / total_q
recall_5_score = recall_5_count / total_q

print(f"\n===== Evaluation Results =====")
print(f"Total Questions: {total_q}")
print(f"Accuracy (Containment logic): {accuracy_score:.2%}")
print(f"Recall@1: {recall_1_score:.2%}")
print(f"Recall@5: {recall_5_score:.2%}")

json_filename = "NLP_HW4_NTHU_114064558_reverse.json"

with open(json_filename, "w", encoding="utf-8") as f:
    json.dump(final_results, f, ensure_ascii=False, indent=4)

print(f"結果已儲存至 {json_filename}")

query = "How much of a day do cats spend sleeping?"

fake_docs = [
    Document(
        page_content="BREAKING NEWS: New scientific studies conclusively prove that cats NEVER sleep. They remain awake 24 hours a day to monitor their human companions.",
        metadata={"id": 999}
    )
]

print("1. Fake ONLY:")
try:
    fake_context_str = fake_docs[0].page_content

    prompt_input = {"context": fake_context_str, "input": query}
    formatted_prompt = prompt.invoke(prompt_input)
    fake_response = llm.invoke(formatted_prompt)

    print(f"Model Answer: {fake_response}")
except Exception as e:
    print(f"Error: {e}")

print("2. Real + Fake:")
try:
    real_retrieved_docs = retriever.invoke(query)

    mixed_docs = real_retrieved_docs + fake_docs

    mixed_context_str = "\n\n".join([d.page_content for d in mixed_docs])

    prompt_input = {"context": mixed_context_str, "input": query}
    formatted_prompt = prompt.invoke(prompt_input)
    mixed_response = llm.invoke(formatted_prompt)

    print(f"Model Answer: {mixed_response}")

except Exception as e:
    print(f"Error: {e}")

print("3. Fake + Real:")
try:
    real_retrieved_docs = retriever.invoke(query)

    mixed_docs = fake_docs + real_retrieved_docs

    mixed_context_str = "\n\n".join([d.page_content for d in mixed_docs])

    prompt_input = {"context": mixed_context_str, "input": query}
    formatted_prompt = prompt.invoke(prompt_input)
    mixed_response = llm.invoke(formatted_prompt)

    print(f"Model Answer: {mixed_response}")

except Exception as e:
    print(f"Error: {e}")

import re
from tqdm import tqdm
from langchain_core.documents import Document
from langchain.retrievers import ContextualCompressionRetriever
from langchain_core.runnables import RunnablePassthrough

def clean_xml_output_v2(text):
    if "ANSWER:" in text:
        text = text.split("ANSWER:")[-1]

    text = re.sub(r'</?doc[^>]*>', '', text)
    text = re.sub(r'(?i)^(the answer is|answer:|answer)\s+', '', text.strip())
    return text.strip().rstrip('.')

xml_system_prompt = (
    "You are a strict data extraction assistant.\n"
    "Task: Answer the question by copying the exact phrase from the provided context.\n"
    "\n"
    "Rules:\n"
    "1. DIRECT QUOTE: Copy the answer exactly as written in the text. Do not rephrase.\n"
    "2. INCLUDE QUALIFIERS: If the text says 'about', 'approx', 'up to', or units like 'mph', KEEP THEM.\n"
    "3. NO MATH: If text says 'Two thirds', do NOT write '16 hours'. Copy 'Two thirds'.\n"
    "4. FOCUS ON CATS: Ignore information about dogs or humans.\n"
    "5. LISTS: If listing items (like colors), do NOT use commas. Just use spaces (e.g. 'Blue green red').\n"
    "6. LENGTH: Keep it under 3 words.\n"
    "\n"
    "Context:\n{context}"
)

xml_prompt = ChatPromptTemplate.from_messages([
    ("system", xml_system_prompt),
    ("human", "{input}")
])

def join_docs(docs):
    return "\n\n".join([d.page_content for d in docs])

xml_chain = (
    {
        "context": rerank_retriever | join_docs,
        "input": RunnablePassthrough()
    }
    | xml_prompt
    | llm
)

xml_correct = 0
xml_recall_1 = 0
xml_recall_5 = 0
debug_count = 0

for i, query in tqdm(enumerate(queries), total=len(queries)):
    try:
        retrieved_docs = rerank_retriever.invoke(query)

        response = xml_chain.invoke(query)
        pred = response
        gt_answer = answers[i]

        target_doc_id = i
        retrieved_ids = [doc.metadata.get("id") for doc in retrieved_docs]
        if len(retrieved_ids) > 0 and retrieved_ids[0] == target_doc_id:
            xml_recall_1 += 1
        if target_doc_id in retrieved_ids:
            xml_recall_5 += 1

        pred_clean = clean_xml_output_v2(pred)

        if normalize_answer(gt_answer) in normalize_answer(pred_clean):
            xml_correct += 1
        else:
            if debug_count < 3:
                print(f"\n[Fail Case {i}]")
                print(f"Q: {query}")
                print(f"Pred (Raw): {pred}")
                print(f"Pred (Clean): {pred_clean}")
                print(f"GT: {gt_answer}")
                debug_count += 1

    except Exception as e:
        print(f"Error: {e}")

print(f"\n===== XML Format Results =====")
print(f"Accuracy: {xml_correct / len(queries):.2%}")
print(f"Recall@1: {xml_recall_1 / len(queries):.2%}")
print(f"Recall@5: {xml_recall_5 / len(queries):.2%}")

import json
import re
from tqdm import tqdm
from langchain_core.documents import Document
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_core.runnables import RunnablePassthrough

docs_json = []
for i, content in enumerate(refs):
    clean_content = content.replace("\n", " ")

    json_obj = {"content": clean_content}
    json_str = json.dumps(json_obj, ensure_ascii=False)

    docs_json.append(Document(page_content=json_str, metadata={"id": i}))

vector_store_json = Chroma.from_documents(
    documents=docs_json,
    embedding=embeddings_model
)


def clean_json_output_v2(text):
    if "ANSWER:" in text:
        text = text.split("ANSWER:")[-1]

    text = text.replace('{"content":', '').replace('}', '').replace('"', '')
    text = re.sub(r'(?i)^(the answer is|answer:|answer)\s+', '', text.strip())

    return text.strip().rstrip('.')

json_system_prompt = (
    "You are a strict data extraction assistant.\n"
"Task: Answer the question by copying the exact phrase from the provided context.\n"
"\n"
"Rules:\n"
"1. DIRECT QUOTE: Copy the answer exactly as written in the text. Do not rephrase.\n"
"2. INCLUDE QUALIFIERS: If the text says 'about', 'approx', 'up to', or units like 'mph', KEEP THEM.\n"
"3. NO MATH: If text says 'Two thirds', do NOT write '16 hours'. Copy 'Two thirds'.\n"
"4. FOCUS ON CATS: Ignore information about dogs or humans.\n"
"5. LISTS: If listing items (like colors), do NOT use commas. Just use spaces (e.g. 'Blue green red').\n"
"6. LENGTH: Keep it under 3 words.\n"
"\n"
"Context:\n{context}"
)

json_prompt = ChatPromptTemplate.from_messages([
    ("system", json_system_prompt),
    ("human", "{input}")
])

def join_docs(docs):
    return "\n\n".join([d.page_content for d in docs])

json_chain = (
    {
        "context": rerank_retriever | join_docs,
        "input": RunnablePassthrough()
    }
    | json_prompt
    | llm
)

json_correct = 0
json_recall_1 = 0
json_recall_5 = 0
debug_count = 0

for i, query in tqdm(enumerate(queries), total=len(queries)):
    try:
        retrieved_docs = rerank_retriever.invoke(query)
        response = json_chain.invoke(query)
        pred = response
        gt_answer = answers[i]

        target_doc_id = i
        retrieved_ids = [doc.metadata.get("id") for doc in retrieved_docs]
        if len(retrieved_ids) > 0 and retrieved_ids[0] == target_doc_id:
            json_recall_1 += 1
        if target_doc_id in retrieved_ids:
            json_recall_5 += 1

        pred_clean = clean_json_output_v2(pred)

        if normalize_answer(gt_answer) in normalize_answer(pred_clean):
            json_correct += 1
        else:
            if debug_count < 3:
                print(f"\n[Fail Case {i}]")
                print(f"Q: {query}")
                print(f"Pred (Raw): {pred}")
                print(f"Pred (Clean): {pred_clean}")
                print(f"GT: {gt_answer}")
                debug_count += 1

    except Exception as e:
        print(f"Error at {i}: {e}")

print(f"\n===== JSON Format Results =====")
print(f"Accuracy: {json_correct / len(queries):.2%}")
print(f"Recall@1: {json_recall_1 / len(queries):.2%}")
print(f"Recall@5: {json_recall_5 / len(queries):.2%}")


