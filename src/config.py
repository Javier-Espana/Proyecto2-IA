import os
from dotenv import load_dotenv
from typing import Dict, Any

# Cargar variables de entorno
load_dotenv()

# ----------------------------
# Configuración de APIs
# ----------------------------

class APIConfig:
    """Configuración de servicios externos"""
    PINECONE = {
        'api_key': os.getenv("PINECONE_API_KEY"),
        'environment': os.getenv("PINECONE_ENVIRONMENT"),
        'index_name': os.getenv("PINECONE_INDEX_NAME", "tech-assistant")
    }
    
    OPENAI = {
        'api_key': os.getenv("OPENAI_API_KEY"),
        'model_name': os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
        'temperature': float(os.getenv("OPENAI_TEMPERATURE", 0.3)),
        'max_tokens': int(os.getenv("OPENAI_MAX_TOKENS", 1000))
    }

# ----------------------------
# Configuración de la Aplicación
# ----------------------------

class AppConfig:
    """Configuración general de la aplicación"""
    # Configuración de documentos
    DOCUMENT_SETTINGS = {
        'chunk_size': int(os.getenv("DOC_CHUNK_SIZE", 1000)),
        'chunk_overlap': int(os.getenv("DOC_CHUNK_OVERLAP", 200)),
        'allowed_extensions': ['.txt', '.pdf', '.md'],
        'default_namespace': os.getenv("DEFAULT_NAMESPACE", "technical")
    }
    
    # Configuración de búsqueda
    SEARCH_SETTINGS = {
        'max_results': int(os.getenv("SEARCH_MAX_RESULTS", 3)),
        'score_threshold': float(os.getenv("SEARCH_SCORE_THRESHOLD", 0.7))
    }
    
    # Configuración de UI
    UI_SETTINGS = {
        'page_title': "Asistente Técnico",
        'page_icon': "🔧",
        'layout': "centered",
        'initial_sidebar_state': "expanded"
    }

# ----------------------------
# Validación de Configuración
# ----------------------------

def validate_config() -> Dict[str, Any]:
    """Verifica que las configuraciones obligatorias estén presentes"""
    errors = {}
    
    if not APIConfig.PINECONE['api_key']:
        errors['pinecone'] = "Falta PINECONE_API_KEY en .env"
    
    if not APIConfig.OPENAI['api_key']:
        errors['openai'] = "Falta OPENAI_API_KEY en .env"
    
    if not APIConfig.PINECONE['index_name']:
        errors['index'] = "Falta PINECONE_INDEX_NAME en .env"
    
    return errors

# ----------------------------
# Configuración por Entorno
# ----------------------------

class ConfigManager:
    """Gestor centralizado de configuración"""
    
    @staticmethod
    def get_embedding_model() -> Dict[str, Any]:
        return {
            'model_name': 'text-embedding-ada-002',
            'dimensions': 1536
        }
    
    @staticmethod
    def get_pinecone_index_config() -> Dict[str, Any]:
        return {
            'dimension': 1536,
            'metric': 'cosine',
            'pod_type': 'p1' if os.getenv("ENVIRONMENT") == 'production' else 'starter'
        }
    
    @staticmethod
    def get_llm_chain_config() -> Dict[str, Any]:
        return {
            'verbose': bool(os.getenv("DEBUG_MODE", False)),
            'max_retries': 3,
            'request_timeout': 30
        }

# ----------------------------
# Inicialización
# ----------------------------

# Validar config al importar
config_errors = validate_config()
if config_errors:
    raise ValueError(f"Errores de configuración: {config_errors}")