# Proyecto2-IA
## Instrucciones para ejecutar
### Entorno virtual
Crear entorno virtual
```bash
python -m venv venv
```
Activar el entorno virtual
```bash
venv\Scripts\activate
```
Instalar las dependecias
```bash
pip install -r requirements.txt
```
### Crear archivo .env
```bash
OPENAI_API_KEY=tu-clave-openai-aquí
PINECONE_API_KEY=tu-clave-pinecone-aquí
PINECONE_REGION=us-east-1
PINECONE_INDEX_NAME=db

DOC_CHUNK_SIZE=1000
DOC_CHUNK_OVERLAP=200
DEFAULT_NAMESPACE=technical

SEARCH_MAX_RESULTS=3
SEARCH_SCORE_THRESHOLD=0.7

ENVIRONMENT=development
DEBUG_MODE=True
```
### Ejecutar Streamlit
```bash
streamlit run src/main.py
```
