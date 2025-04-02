from sentence_transformers import SentenceTransformer


def get_detailed_instruct(task_description: str, query: str) -> str:
    return f"Instruct: {task_description}\nQuery: {query}"


# Each query must come with a one-sentence instruction that describes the task
task_query = "Represent the characteristics of the house that the user is looking for"
queries = [
    get_detailed_instruct(task_query, "Busco piso con terraza"),
]

task_listing = "Represent the characteristics of the house description"
documents = [
    "Hamburguesa con queso",
    "Buen piso en valencia.",
    "Primer piso con gran terraza exterior a plaza peatonal. Para reformar como vivienda o como despacho, tras acristalar parte de la terraza actualmente tiene 200 m2 más 50 en 2 terrazas. Buena situación y buen edificio. 2 orientaciones: Sur y Norte. Actualmente academia (sin cocina).",
    "Piso en planta baja exterior distribuido en cuatro habitaciones (una de ellas con vestidor), dos baños, salón y cocina con terraza. Calefacción de gas natural individual.",
    "Ático de 40 m2 + 50 metros de terraza. Muy luminoso. Salón con cocina americana, 1 dormitorio y 1 baño. Piscina en la comunidad",
    """ Buskpiso ofrece este coqueto piso para entrar a vivir con una reforma de menos de doce años. 

        El salón dispone de grandes ventanales que lo inundan de luz por las mañanas. Las dos habitaciones son amplias y luminosas, dando a gran patio interior, ideal para un perfecto descanso. Ambas disponen de armarios empotrados de hechura reciente. La cocina, amueblada en un color blanco que nunca pasa de moda y bonito suelo de gres. El agradable cuarto de baño dispone de ducha de hidromasaje para relajar los músculos antes de ir al trabajo.

        El piso dispone de suelo de parquet, ventanas con carpintería de climalit, terraza, caldera de menos de un año, aire acondicionado en las habitaciones y el salón.

        Tiene alarma instalada.

        Totalmente amueblado y listo para entrar a vivir.

        Llámanos para hacer una visita.

    """,
]
documents = [get_detailed_instruct(task_listing, doc) for doc in documents]
input_texts = queries + documents

model = SentenceTransformer("intfloat/multilingual-e5-large-instruct")

embeddings = model.encode(
    input_texts, convert_to_tensor=True, normalize_embeddings=True
)
scores = (embeddings[:1] @ embeddings[1:].T) * 100
print(scores)
# [[91.92853546142578, 67.5802993774414], [70.38143157958984, 92.13307189941406]]
