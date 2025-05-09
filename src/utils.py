import os
from typing import List, Dict
from langchain.schema import Document
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import tiktoken

# Cargar variables de entorno
load_dotenv()

# ----------------------------
# Funciones de Procesamiento
# ----------------------------

def count_tokens(text: str) -> int:
    """Cuenta tokens para un texto usando el encoding de OpenAI"""
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    return len(encoding.encode(text))

def split_text(text: str, max_tokens: int = 2000) -> List[str]:
    """
    Divide texto en chunks respetando límite de tokens.
    Args:
        text: Texto a dividir
        max_tokens: Límite de tokens por chunk (default 2000)
    Returns:
        Lista de chunks de texto
    """
    paragraphs = text.split('\n\n')
    chunks = []
    current_chunk = []
    current_count = 0

    for para in paragraphs:
        para_tokens = count_tokens(para)
        
        if current_count + para_tokens > max_tokens:
            chunks.append('\n\n'.join(current_chunk))
            current_chunk = [para]
            current_count = para_tokens
        else:
            current_chunk.append(para)
            current_count += para_tokens
    
    if current_chunk:
        chunks.append('\n\n'.join(current_chunk))
    
    return chunks

# ----------------------------
# Generación de Respuestas
# ----------------------------

def generate_response(query: str, context_docs: List[Document]) -> Dict:
    """
    Genera respuesta usando contexto relevante y OpenAI.
    Args:
        query: Pregunta del usuario
        context_docs: Lista de documentos relevantes (LangChain Documents)
    Returns:
        Dict con respuesta y metadatos
    """
    # Combinar contexto en un solo texto
    context_text = "\n\n".join([doc.page_content for doc in context_docs])
    sources = list(set([doc.metadata.get('source', '') for doc in context_docs]))
    
    # Plantilla para el prompt
    prompt_template = """
    Eres un asistente técnico especializado. Responde la pregunta del usuario 
    usando SOLO el contexto proporcionado. Sé preciso y técnico pero claro.
    
    Si no sabes la respuesta, di simplemente "No tengo información suficiente".
    
    Contexto:
    {context}
    
    Pregunta: {question}
    Respuesta:"""
    
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    
    # Configurar modelo y cadena
    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        temperature=0.3,  # Menor temperatura = más preciso/focalizado
        max_tokens=1000
    )
    
    chain = LLMChain(llm=llm, prompt=prompt)
    
    # Generar respuesta
    response = chain.run({
        "context": context_text,
        "question": query
    })
    
    return {
        "answer": response.strip(),
        "sources": sources,
        "context_used": context_text[:2000] + "..." if len(context_text) > 2000 else context_text
    }

# ----------------------------
# Validación de Entrada
# ----------------------------

def validate_query(query: str) -> bool:
    """Valida que la consulta no esté vacía y sea razonable"""
    if not query or len(query.strip()) < 5:
        return False
    if count_tokens(query) > 300:
        return False
    return True

def clean_filename(filename: str) -> str:
    """Limpia nombres de archivo para usarlos como IDs en Pinecone"""
    return os.path.splitext(filename)[0].replace(" ", "_").lower()