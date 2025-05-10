from pinecone import Pinecone
from dotenv import load_dotenv
import os

load_dotenv()

Pine_Cone_Key = os.getenv("PINECONE_API_KEY")

pc = Pinecone(api_key=Pine_Cone_Key)

index_name = "demo3-index"
if not pc.has_index(index_name):
    pc.create_index_for_model(
        name=index_name,
        cloud="aws",
        region="us-east-1",
        embed={
            "model":"llama-text-embed-v2",
            "field_map":{"text":"chunk_text"},
        }
    )

records = [
    {"_id": "rec1", "chunk_text": "La biodiversidad se refiere a la variedad de formas de vida en la Tierra.","category": "biology"},
    {"_id": "rec2", "chunk_text": "Los genes son segmentos de ADN que codifican características hereditarias.","category": "biology"},
    {"_id": "rec3", "chunk_text": "El sistema inmunológico protege al cuerpo contra infecciones y enfermedades.","category": "biology"},
    {"_id": "rec4", "chunk_text": "Los ecosistemas son comunidades de organismos que interactúan entre sí y con su entorno.","category": "biology"},
    {"_id": "rec5", "chunk_text": "Los ribosomas son responsables de la síntesis de proteínas dentro de las células.","category": "biology"},
    {"_id": "rec6", "chunk_text": "La evolución explica cómo las especies cambian con el tiempo a través de la selección natural.","category": "biology"},
    {"_id": "rec7", "chunk_text": "Las mitocondrias son orgánulos celulares encargados de producir energía en forma de ATP.","category": "biology"},
    {"_id": "rec8", "chunk_text": "La fotosíntesis es el proceso mediante el cual las plantas convierten la luz solar en energía química.","category": "biology"},
    {"_id": "rec9", "chunk_text": "El ADN (ácido desoxirribonucleico) contiene la información genética de los organismos.","category": "biology"},
    {"_id": "rec10", "chunk_text": "La célula es la unidad estructural y funcional básica de todos los seres vivos.","category": "biology"},
    {"_id": "rec11", "chunk_text": "El río Misisipi es uno de los ríos más largos de América del Norte y atraviesa diez estados de EE. UU.","category": "geography"},
    {"_id": "rec12", "chunk_text": "Islandia es una nación insular nórdica famosa por sus volcanes, géiseres y aguas termales.","category": " 12"},
    {"_id": "rec13", "chunk_text": "El desierto de Gobi se encuentra en el norte de China y el sur de Mongolia, y es conocido por su clima extremo.","category": "geography"},
    {"_id": "rec14", "chunk_text": "Los Andes se extienden por el borde occidental de América del Sur y son la cordillera continental más larga.","category": "geography"},
    {"_id": "rec15", "chunk_text": "La Antártida es el continente más frío, seco y ventoso de la Tierra.","category": "geography"},
    {"_id": "rec16", "chunk_text": "Groenlandia es la isla más grande del mundo y está cubierta en su mayoría por hielo.","category": "geography"},
    {"_id": "rec17", "chunk_text": "El lago Baikal, en Rusia, es el lago de agua dulce más profundo y antiguo del mundo.","category": "geography"},
    {"_id": "rec18", "chunk_text": "El Himalaya es una extensa cadena montañosa en Asia que alberga muchas de las cumbres más altas del planeta.","category": "geography"},
    {"_id": "rec19", "chunk_text": "La Gran Barrera de Coral, situada frente a la costa de Australia, es el sistema de arrecifes de coral más grande del mundo.","category": "geography"},
    {"_id": "rec20", "chunk_text": "El río Nilo es el río más largo del mundo y atraviesa el noreste de África.","category": "geography"},
    {"_id": "rec21", "chunk_text": "La Ilustración fue un movimiento intelectual que promovió la razón y el conocimiento en el siglo XVIII.","category": "historia"},
    {"_id": "rec22", "chunk_text": "Venezuela fue el primer país sudamericano en declarar su independencia en 1811, con Simón Bolívar liderando la lucha contra el dominio español.","category": "historia"},
    {"_id": "rec23", "chunk_text": "El Salvador participó en la independencia centroamericana en 1821, y luego enfrentó guerras civiles que culminaron en los Acuerdos de Paz de 1992.","category": "historia"},
    {"_id": "rec24", "chunk_text": "Nicaragua, tras su independencia en 1821, vivió conflictos internos entre liberales y conservadores que marcaron su historia del siglo XIX.","category": "historia"},
    {"_id": "rec25", "chunk_text": "El Imperio Inca fue uno de los más grandes y avanzados de América precolombina.","category": "historia"},
    {"_id": "rec26", "chunk_text": "La Revolución Mexicana comenzó en 1910 y transformó profundamente el país.","category": "historia"},
    {"_id": "rec27", "chunk_text": "Simón Bolívar lideró las luchas de independencia en varios países de América del Sur.","category": "historia"},
    {"_id": "rec28", "chunk_text": "La Revolución Industrial comenzó en Inglaterra a finales del siglo XVIII y se extendió por todo el mundo.","category": "historia"},
    {"_id": "rec29", "chunk_text": "El Imperio Romano alcanzó su máxima extensión territorial bajo Trajano en el año 117 d.C.","category": "historia"},
    {"_id": "rec30", "chunk_text": "Las Pirámides de Giza fueron construidas alrededor del 2580-2560 a.C. durante la Cuarta Dinastía del Antiguo Egipto.","category": "historia"},
    {"_id": "rec31", "chunk_text": "Los compuestos orgánicos contienen carbono e hidrógeno como elementos principales.","category": "chemistry"},
    {"_id": "rec32", "chunk_text": "Una reacción exotérmica libera energía en forma de calor.","category": "chemistry"},
    {"_id": "rec33", "chunk_text": "Las soluciones químicas se componen de un soluto disuelto en un disolvente.","category": "chemistry"},
    {"_id": "rec34", "chunk_text": "Los catalizadores aumentan la velocidad de las reacciones químicas sin consumirse.","category": "chemistry"},
    {"_id": "rec35", "chunk_text": "El pH mide la acidez o alcalinidad de una solución en una escala de 0 a 14.","category": "chemistry"},
    {"_id": "rec36", "chunk_text": "La ley de conservación de la masa indica que la materia no se crea ni se destruye.","category": "chemistry"},
    {"_id": "rec37", "chunk_text": "El enlace covalente implica el compartimiento de electrones entre átomos.","category": "chemistry"},
    {"_id": "rec38", "chunk_text": "La reacción de neutralización ocurre entre un ácido y una base para formar sal y agua.","category": "chemistry"},
    {"_id": "rec39", "chunk_text": "La tabla periódica organiza los elementos según su número atómico.","category": "chemistry"},
    {"_id": "rec40", "chunk_text": "El agua es un compuesto formado por dos átomos de hidrógeno y uno de oxígeno.","category": "chemistry"},
    {"_id": "rec41", "chunk_text": "El automóvil, desarrollado en gran parte por Karl Benz en 1885, revolucionó el transporte personal y comercial.","category": "Inventos"},
    {"_id": "rec42", "chunk_text": "La máquina de vapor, perfeccionada en el siglo XVIII, impulsó la Revolución Industrial.","category": "Inventos"},
    {"_id": "rec43", "chunk_text": "La computadora personal, popularizada en los años 70 y 80, permitió el acceso individual a la tecnología informática.","category": "Inventos"},
    {"_id": "rec44", "chunk_text": "El internet, desarrollado a finales del siglo XX, transformó la comunicación, el comercio y el acceso a la información.","category": "Inventos"},
    {"_id": "rec45", "chunk_text": "La penicilina, descubierta por Alexander Fleming en 1928, revolucionó la medicina moderna.","category": "Inventos"},
    {"_id": "rec46", "chunk_text": "El avión, inventado por los hermanos Wright en 1903, cambió para siempre la forma en que viajamos.","category": "Inventos"},
    {"_id": "rec47", "chunk_text": "La rueda, uno de los inventos más antiguos, fue clave para el transporte y la mecánica.","category": "Inventos"},
    {"_id": "rec48", "chunk_text": "La bombilla eléctrica, perfeccionada por Thomas Edison en 1879, iluminó hogares y calles.","category": "Inventos"},
    {"_id": "rec49", "chunk_text": "El teléfono, desarrollado por Alexander Graham Bell en 1876, permitió la comunicación instantánea a distancia.","category": "Inventos"},
    {"_id": "rec50", "chunk_text": "La imprenta, inventada por Johannes Gutenberg en el siglo XV, revolucionó la difusión del conocimiento.","category": "Inventos"}
]   

dense_index = pc.Index(index_name)

dense_index.upsert_records("demo3", records)

import time

time.sleep(5)

stats = dense_index.describe_index_stats()
print(f"Stats: {stats}")


query= "cual es el mejor invento de la historia"
results = dense_index.search(
    namespace="demo3",
    query={
        "top_k": 5,
        "inputs":{ 'text':query}
    }
)
for i in results['result']['hits']:
    print(f"id: {i['_id']:<5} | score: {round(i['_score'],2):<5} | text: {i['fields']['chunk_text']:<50} | category: {i['fields']['category']:<10}")