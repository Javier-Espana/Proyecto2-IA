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
## Descripción de los módulos

- **[`src/main.py`](src/main.py):**  
  Interfaz principal de usuario usando Streamlit. Permite cargar documentos, realizar consultas técnicas y visualizar respuestas generadas por IA. Gestiona el historial de chat y la interacción con el usuario.

- **[`src/vector_db.py`](src/vector_db.py):**  
  Encargado de la gestión de la base de datos vectorial Pinecone. Permite cargar documentos, dividirlos en fragmentos, generar embeddings, realizar búsquedas semánticas y administrar namespaces.

- **[`src/utils.py`](src/utils.py):**  
  Contiene utilidades para el procesamiento de texto, conteo de tokens, validación de consultas, generación de respuestas usando modelos de lenguaje y normalización de nombres de archivos.

- **[`src/config.py`](src/config.py):**  
  Centraliza la configuración de la aplicación y la validación de variables de entorno. Define parámetros para la conexión a APIs externas, la interfaz y el procesamiento de documentos.

---

## ¿Qué aprendimos al desarrollar el asistente?

Durante el desarrollo de este asistente técnico, aprendimos a integrar múltiples servicios de IA y cloud, como OpenAI y Pinecone, para construir una solución de búsqueda semántica sobre documentos personalizados. Profundizamos en el manejo de embeddings, la división eficiente de textos y la importancia de una buena gestión de configuración y seguridad (uso de `.env`). Además, reforzamos habilidades en el desarrollo de interfaces interactivas con Streamlit y en la estructuración de proyectos modulares y escalables en Python.

El trabajo en equipo nos permitió distribuir tareas, compartir conocimientos sobre procesamiento de lenguaje natural y enfrentar juntos los retos de integración y despliegue. Individualmente, cada miembro mejoró su comprensión sobre arquitecturas de asistentes conversacionales y la aplicación práctica de modelos de lenguaje en soluciones reales.

---