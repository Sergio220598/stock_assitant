# agent_rag.py
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from functools import partial
import os

def agent_rag(query: str, k: int = 3) -> str: #Recupera los 3 fragmentos más relevantes.
    """
    Recupera información relevante desde la carpeta 'documents'
    y responde a la pregunta usando RAG simple.
    """
    carpeta = "documents"
    if not os.path.exists(carpeta):
        return "No se encontró la carpeta 'documents'."

    # 1️⃣ Cargar documentos (PDF y TXT)
    docs = []
    try:
        #docs += DirectoryLoader(carpeta, glob="**/*.pdf", loader_cls=PyPDFLoader).load()

        utf8_loader = partial(TextLoader, encoding="utf-8", autodetect_encoding=True)
        docs += DirectoryLoader(carpeta, glob="**/*.txt", loader_cls=utf8_loader).load()
    except Exception as e:
        return f"No se pudieron cargar los documentos: {e}"

    if not docs:
        return "No se encontraron documentos válidos en la carpeta."

    # 2️⃣ Dividir el texto en fragmentos
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    # 3️⃣ Crear embeddings y FAISS temporal
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    # 4️⃣ Buscar los fragmentos más relevantes
    resultados = retriever.invoke(query)

    # 5️⃣ Combinar el contexto y generar respuesta
    contexto = "\n".join([doc.page_content for doc in resultados])
    llm = ChatOpenAI(model="gpt-5", temperature=0.3)

    prompt = f"""
    Usa el siguiente contexto para responder de forma clara y breve (máx. 5 oraciones):

    Contexto:
    {contexto}

    Pregunta: {query}
    """

    respuesta = llm.invoke(prompt)
    return respuesta.content


#print(agent_rag("explicame que es el syp500"))