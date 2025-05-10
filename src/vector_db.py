import os
import hashlib
from typing import List, Optional
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.schema import Document
from pinecone import Pinecone as PineconeClient, ServerlessSpec
from dotenv import load_dotenv
from utils import clean_filename, split_text

# Cargar variables de entorno
load_dotenv()

# ----------------------------
# Configuración inicial
# ----------------------------

def initialize_pinecone() -> PineconeClient:
    """Inicializa la conexión con Pinecone y devuelve el cliente"""
    return PineconeClient(api_key=os.getenv("PINECONE_API_KEY"))

# ----------------------------
# Carga y procesamiento de documentos
# ----------------------------

def load_document(file_path: str) -> List[Document]:
    """Carga un documento y lo divide en chunks"""
    if file_path.endswith('.pdf'):
        loader = PyPDFLoader(file_path)
    else:
        loader = TextLoader(file_path, encoding='utf-8')
    
    raw_docs = loader.load()
    
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

def create_or_get_index(pc: PineconeClient, index_name: Optional[str] = None) -> None:
    """Crea un nuevo índice o verifica si existe usando has_index()."""
    index_name = index_name or os.getenv("PINECONE_INDEX_NAME")

    # Verifica si el índice existe
    if not pc.has_index(index_name):
        pc.create_index(
            name=index_name,
            dimension=1536,           # Ajusta según tu embedding
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"   # Región compatible con tu plan
            )
        )
        print(f"Índice '{index_name}' creado")
    else:
        print(f"Índice '{index_name}' ya existe")

def load_documents_to_pinecone(file_paths: List[str], namespace: str = "technical") -> None:
    """Procesa y carga documentos a Pinecone evitando duplicados"""
    pc = initialize_pinecone()
    index_name = os.getenv("PINECONE_INDEX_NAME")
    create_or_get_index(pc, index_name)
    
    embeddings = OpenAIEmbeddings()
    all_docs = []
    
    # Preparamos el índice para la verificación de duplicados
    index = pc.Index(index_name)
    
    for file_path in file_paths:
        try:
            docs = load_document(file_path)
            for doc in docs:
                doc.metadata["source"] = os.path.basename(file_path)
                doc.metadata["id"] = generate_document_id(doc.page_content, file_path)
            all_docs.extend(docs)
        except Exception as e:
            print(f"Error procesando {file_path}: {str(e)}")
    
    # Convertir documentos a vectores
    if all_docs:
        vectors = []
        for doc in all_docs:
            embedding = embeddings.embed_query(doc.page_content)
            # Incluimos explícitamente el texto en la metadata:
            meta = {
                "id": doc.metadata["id"],
                "source": doc.metadata["source"],
                "text": doc.page_content,           # <— aquí!
            }
            vectors.append({
                "id": doc.metadata["id"],
                "values": embedding,
                "metadata": meta
            })
        
        # Consulta de IDs existentes
        existing_ids = set()  # Usamos un set para optimizar la búsqueda
        for vector in vectors:
            try:
                response = index.query(
                    vector=vector["values"],
                    top_k=1,
                    namespace=namespace
                )
                if response["matches"]:
                    existing_ids.add(vector["id"])  # El ID ya existe
            except Exception as e:
                print(f"Error al consultar el vector: {e}")
        
        # Filtra los vectores duplicados
        new_vectors = [vec for vec in vectors if vec["id"] not in existing_ids]
        
        # Si hay nuevos vectores, realiza el upsert
        if new_vectors:
            index.upsert(vectors=new_vectors, namespace=namespace)
            print(f"{len(new_vectors)} nuevos documentos cargados en namespace '{namespace}'")
        else:
            print("No se encontraron documentos nuevos para cargar.")


# ----------------------------
# Búsqueda semántica
# ----------------------------

def query_similar_docs(query: str, k: int = 3, namespace: str = "technical") -> List[Document]:
    """Realiza búsqueda semántica en Pinecone"""
    pc = initialize_pinecone()
    index_name = os.getenv("PINECONE_INDEX_NAME")
    embeddings = OpenAIEmbeddings()
    
    index = pc.Index(index_name)
    
    # Generar el embedding de la consulta
    query_vector = embeddings.embed_query(query)
    
    # Realizar la consulta en Pinecone
    result = index.query(
        vector=query_vector,
        top_k=k,
        namespace=namespace,
        include_metadata=True
    )
    
    # Convertir los resultados en documentos
    documents = []
    for match in result['matches']:
        doc = Document(
            page_content=match['metadata']['text'],
            metadata=match['metadata']
        )
        documents.append(doc)
    
    return documents


# ----------------------------
# Mantenimiento
# ----------------------------

def delete_namespace(namespace: str = "technical") -> None:
    """Elimina todos los vectores en un namespace"""
    pc = initialize_pinecone()
    index = pc.Index(os.getenv("PINECONE_INDEX_NAME"))
    index.delete(delete_all=True, namespace=namespace)
    print(f"Namespace '{namespace}' eliminado")

def list_namespaces() -> List[str]:
    """Lista todos los namespaces en el índice"""
    pc = initialize_pinecone()
    index = pc.Index(os.getenv("PINECONE_INDEX_NAME"))
    stats = index.describe_index_stats()
    return list(stats["namespaces"].keys())
