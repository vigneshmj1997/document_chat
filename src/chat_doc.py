"""
Reader-Retriever Architecture for Document Chat

Retriever Model:
Finds potential locations of answers in the document based on sentence similarity.

Reader Model:
A Decoder-based model that elaborates on retrieved text to extract the final answer.

Document Chat Pipeline:
1. Question is passed to the Retriever.
2. Retriever retrieves candidate passages from the document.
3. Candidate passages are fed into the Reader.
4. Reader generates the final answer based on the question and passages.

Note: This architecture is commonly used in NLP applications like chatbots and information retrieval systems.
"""

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from string import Template
from langchain.memory import ConversationBufferMemory
from typing import List
from gpt4all import GPT4All
import argparse
import pandas as pd
from tqdm import tqdm


def load_and_chunk_document(document_path: str) -> List:
    """
    Loads a document from 'document_path' and splits it into smaller chunks.

    Parameters:
        document_path (str): File path of the document to be loaded and chunked.

    Returns:
        List[str]: List of smaller document chunks as strings.

    Description:
    This function loads the content of the document from 'document_path' and breaks it
    into smaller chunks for efficient processing. It uses 'RecursiveCharacterTextSplitter'
    to split the document, with default 'chunk_size=100' and 'chunk_overlap=10'.

    Example Usage:
    chunks = load_and_chunk_document('example.txt')
    for chunk in chunks:
        print(chunk)
    """
    document = open(document_path).read()
    splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
    chunks = splitter.create_documents([document])

    return chunks


def get_chat_prompt(context: str, query: str, history: str) -> str:
    """
    Generates a chat prompt with context, query, and history.
    """
    DEFAULT_CHAT_PROMPT = """
    Use the following pieces of context to answer the query at the end.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    I will provide you with our conversation history.

    $context

    History: $history

    Query: $query

    Helpful Answer:
    """
    DEFAULT_CHAT_PROMPT_TEMPLATE = Template(DEFAULT_CHAT_PROMPT)
    return DEFAULT_CHAT_PROMPT_TEMPLATE.substitute(
        context=context, query=query, history=history
    )


def create_embeddings(all_splits: List, multilingual=False) -> Chroma:
    """
    Creats embeddings using huggingface embeddings and stores it in chroma db
    """
    vectorstore = Chroma.from_documents(
        documents=all_splits,
        embedding=HuggingFaceEmbeddings(
            model_name="sentence-transformers/stsb-xlm-r-multilingual"
            if multilingual
            else "sentence-transformers/all-mpnet-base-v2"
        ),
    )
    return vectorstore


def retrieve_from_database(
    input_query: str, vectorstore: Chroma, number_documents: int = 5
):
    """
    Retrieves similar documents from 'vectorstore' based on 'input_query'.
    """
    similar_docs = vectorstore.similarity_search(
        input_query, number_documents=number_documents
    )
    return similar_docs


def get_args():
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("--path", type=str, help="Location of the document")
    parser.add_argument("--test_path", type=str, help="Location of the document")
    parser.add_argument(
        "--multilingual",
        action="store_true",
        help="Specify if the chat supports multilingual queries",
    )
    args = parser.parse_args()
    return args


def chat(args):
    chunks = load_and_chunk_document(args.path)
    vectorstore = create_embeddings(chunks, args.multilingual)
    memory = ConversationBufferMemory()
    gpt4all_model = GPT4All("orca-mini-3b.ggmlv3.q4_0.bin")
    chat_history = memory.load_memory_variables({})["history"]

    if args.test_path:
        # this fuction is written inside cause it has no other dependency
        import sacrebleu

        def calculate_sacrebleu(references, hypotheses):
            """
            Calculate the sacreBLEU score for given references and hypotheses.

            Args:
                references (List[str]): List of reference sentences.
                hypotheses (List[str]): List of hypothesis sentences.

            Returns:
                float: The sacreBLEU score.
            """
            # Calculate the sacreBLEU score
            bleu = sacrebleu.corpus_bleu(hypotheses, [references])
            return bleu.score

        df = pd.read_csv(args.test_path)
        response = {"question": [], "actual_ans": [], "prediucted_ans": []}
        for i in tqdm(len(df)):
            query = df["Question"].iloc[i]
            context = retrieve_from_database(query, vectorstore)
            prompt = get_chat_prompt(context, query, chat_history)
            answer = gpt4all_model.generate(prompt=prompt)
            response["question"] = query
            response["actual_ans"] = df["Ideal Answer"].iloc[i]
            response["prediucted_ans"] = answer

        score = calculate_sacrebleu(response["actual_ans"], response["prediucted_ans"])
        print(f"The bleu score of the prediciton is: {score}")
        exit()

    while True:
        query = str(input("Ask a question (type 'exit' to stop): "))
        if query.lower().find("exit") == -1:
            context = retrieve_from_database(query, vectorstore)
            prompt = get_chat_prompt(context, query, chat_history)
            answer = gpt4all_model.generate(prompt=prompt)
            memory.chat_memory.add_user_message(query)
            memory.chat_memory.add_ai_message(answer)
            print(f"Reply: {answer}")
        else:
            exit()


def main():
    args = get_args()
    chat(args)


if __name__ == "__main__":
    main()
