import pandas as pd
import spacy
from langdetect import detect
from utils import (
    extract_room_info,
    process_text_row,
    extract_price_number,
    extract_area_number,
)
from functools import partial

# Read CSV file

df = pd.read_csv("data/comments-cleaned.csv")
locations = pd.read_csv("data/location-names.csv", sep=";")
locations.columns = ["ASSETID", "municipio", "distrito", "barrio"]

# Drop duplicates
df = df.drop_duplicates("ASSETID")
locations = locations.drop_duplicates("ASSETID")


# Detect language and filter spanish listings
def detect_language(text):
    try:
        return detect(text)
    except:
        return "unknown"


df["language"] = df["TEXT"].apply(detect_language)
df = df.query("language == 'es'")
# Join
df = df.set_index("ASSETID").join(
    locations.set_index("ASSETID"), how="left", on="ASSETID"
)

# Perform NER
nlp = spacy.load("./models/model-best")
# Obtener el attribute_ruler
ruler = nlp.get_pipe("attribute_ruler")

# Definir patrones
patterns = [
    [{"LOWER": "metros"}, {"LOWER": "cuadrados"}],
    [{"LOWER": "m2"}],
    [{"LOWER": "m²"}],
    [{"LOWER": "m^2"}],
]

# Normalizar a "m2"
attrs = {"LEMMA": "m2"}

ruler.add(patterns=patterns, attrs=attrs)

# Process the listing
caracteristicas_validas = {
    "piscina",
    "terraza",
    "aire acondicionado",
    "armario empotrado",
    "balcón",
    "ascensor",
    "calefacción",
}
partial_process_text_row = partial(
    process_text_row, nlp=nlp, caracteristicas_validas=caracteristicas_validas
)
df_results = df["TEXT"].apply(partial_process_text_row).apply(pd.Series)
df_results["precio"] = df_results["precio_detected"].apply(extract_price_number)
df_results["area"] = df_results["area_detected"].apply(extract_area_number)
df = pd.concat([df, df_results], axis=1)

# Save the updated DataFrame to a new CSV file
df_name = "comments_postprocessed.csv"
df.to_csv(f"{df_name}", index=True)

print(f"CSV saved as '{df_name}'.")
