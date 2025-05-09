import os
import hashlib
from typing import List, Optional
from langchain.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.schema import Document
import pinecone
from dotenv import load_dotenv
from utils import clean_filename, split_text

# Cargar variables de entorno
load_dotenv()

# ----------------------------
# Configuración inicial
# ----------------------------

def initialize_pinecone():
    """Inicializa la conexión con Pinecone"""
    pinecone.init(
        api_key=os.getenv("PINECONE_API_KEY"),
        environment=os.getenv("PINECONE_ENVIRONMENT")
    )

# ----------------------------
# Carga y procesamiento de documentos
# ----------------------------

def load_document(file_path: str) -> List[Document]:
    """
    Carga un documento y lo divide en chunks
    Args:
        file_path: Ruta al archivo (PDF o TXT)
    Returns:
        Lista de Documentos de LangChain
    """
    if file_path.endswith('.pdf'):
        loader = PyPDFLoader(file_path)
    else:
        loader = TextLoader(file_path, encoding='utf-8')
    
    raw_docs = loader.load()
    
    # Configurar el splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    
    return text_splitter.split_documents(raw_docs)

def generate_document_id(text: str, source: str) -> str:
    """Genera un ID único para cada chunk basado en su contenido"""
    content_hash = hashlib.md5(text.encode()).hexdigest()[:8]
    source_prefix = clean_filename(source)[:20]
    return f"{source_prefix}-{content_hash}"

# ----------------------------
# Operaciones con Pinecone
# ----------------------------

def create_or_get_index(index_name: Optional[str] = None) -> None:
    """
    Crea un nuevo índice o verifica si existe
    Args:
        index_name: Nombre del índice (si None, usa de .env)
    """
    index_name = index_name or os.getenv("PINECONE_INDEX_NAME")
    
    if index_name not in pinecone.list_indexes():
        # Crear nuevo índice con dimensión para embeddings de OpenAI
        pinecone.create_index(
            name=index_name,
            dimension=1536,  # Dimensión para text-embedding-ada-002
            metric="cosine"
        )
        print(f"Índice {index_name} creado")
    else:
        print(f"Índice {index_name} ya existe")

def load_documents_to_pinecone(file_paths: List[str], namespace: str = "technical") -> None:
    """
    Procesa y carga documentos a Pinecone
    Args:
        file_paths: Lista de rutas a archivos
        namespace: Namespace en Pinecone para estos documentos
    """
    initialize_pinecone()
    index_name = os.getenv("PINECONE_INDEX_NAME")
    create_or_get_index(index_name)
    
    embeddings = OpenAIEmbeddings()
    all_docs = []
    
    for file_path in file_paths:
        try:
            docs = load_document(file_path)
            # Agregar metadatos y IDs únicos
            for doc in docs:
                doc.metadata["source"] = os.path.basename(file_path)
                doc.metadata["id"] = generate_document_id(doc.page_content, file_path)
            all_docs.extend(docs)
        except Exception as e:
            print(f"Error procesando {file_path}: {str(e)}")
    
    if all_docs:
        Pinecone.from_documents(
            all_docs,
            embeddings,
            index_name=index_name,
            namespace=namespace
        )
        print(f"{len(all_docs)} documentos cargados en namespace '{namespace}'")

# ----------------------------
# Búsqueda semántica
# ----------------------------

def query_similar_docs(query: str, k: int = 3, namespace: str = "technical") -> List[Document]:
    """
    Realiza búsqueda semántica en Pinecone
    Args:
        query: Texto de la consulta
        k: Número de resultados a devolver
        namespace: Namespace donde buscar
    Returns:
        Lista de documentos relevantes
    """
    initialize_pinecone()
    index_name = os.getenv("PINECONE_INDEX_NAME")
    embeddings = OpenAIEmbeddings()
    
    vector_store = Pinecone(
        pinecone.Index(index_name),
        embeddings.embed_query,
        "text",  # Nombre de la columna con el texto
        namespace=namespace
    )
    
    return vector_store.similarity_search(query, k=k)

# ----------------------------
# Mantenimiento
# ----------------------------

def delete_namespace(namespace: str = "technical") -> None:
    """Elimina todos los vectores en un namespace"""
    initialize_pinecone()
    index = pinecone.Index(os.getenv("PINECONE_INDEX_NAME"))
    index.delete(delete_all=True, namespace=namespace)
    print(f"Namespace '{namespace}' eliminado")

def list_namespaces() -> List[str]:
    """Lista todos los namespaces en el índice"""
    initialize_pinecone()
    index = pinecone.Index(os.getenv("PINECONE_INDEX_NAME"))
    stats = index.describe_index_stats()
    return list(stats["namespaces"].keys())