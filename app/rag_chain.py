from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate
from app.ingest import get_vector_store
from app.config import GROQ_API_KEY, GROQ_MODEL

def build_rag_chain():
    """
    Builds and returns a ConversationalRetrievalChain.
    This chain has three components wired together:
    1. Retriever — fetches relevant chunks from Pinecone
    2. Memory — remembers previous messages in the session
    3. LLM — Llama 3.2 via Groq generates the final answer
    """

    # --- 1. LLM setup ---
    # ChatGroq connects to the Groq API
    # temperature=0.3 means balanced creativity and reliability
    # Higher temperature (e.g. 0.8) = more creative but less reliable
    llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model=GROQ_MODEL,
    temperature=0.3,
    )

    # --- 2. Retriever setup ---
    # Get our Pinecone vector store and convert it to a retriever
    # k=4 means we fetch the 4 most relevant chunks for each question
    vector_store = get_vector_store()
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 6}
    )

    # --- 3. Memory setup ---
    # ConversationBufferWindowMemory remembers the last k exchanges
    # k=5 means it keeps the last 5 question/answer pairs in context
    # memory_key must match the variable name in our prompt template
    # return_messages=True formats history as chat messages (required for ConversationalRetrievalChain)
    memory = ConversationBufferWindowMemory(
        k=5,
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"  # tells memory which part of the output to store
    )

    # --- 4. Prompt template ---
    # This is the instruction set Llama receives along with every question
    # {context} gets filled with the retrieved chunks
    # {question} gets filled with the user's question
    prompt_template = """You are a helpful assistant answering questions about a document.

Use the context below to answer the question. The context contains excerpts from the document.

Context:
{context}

Question: {question}

Instructions:
- Answer based on the context provided above
- Be specific and detailed in your answer
- If you truly cannot find relevant information in the context, say so briefly
- Do not say "I don't have enough information" if the context clearly contains relevant content

Answer:"""

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    # --- 5. Build the chain ---
    # ConversationalRetrievalChain wires all three components together
    # combine_docs_chain_kwargs passes our custom prompt to the answer generation step
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,   # includes the chunks used in the response
        combine_docs_chain_kwargs={"prompt": prompt}
    )

    return chain


def ask_question(chain, question: str) -> dict:
    """
    Sends a question through the RAG chain and returns
    the answer plus source citations.
    """
    response = chain.invoke({"question": question})

    # Extract the source documents and format them as citations
    sources = []
    for doc in response.get("source_documents", []):
        sources.append({
            "page": doc.metadata.get("page", "unknown"),
            "source": doc.metadata.get("source", "unknown"),
            "content_preview": doc.page_content[:150]  # first 150 chars as preview
        })

    return {
        "answer": response["answer"],
        "sources": sources
    }