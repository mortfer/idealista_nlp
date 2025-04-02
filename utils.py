import re

NUMEROS_ES = {
    "uno": 1,
    "una": 1,
    "un": 1,
    "dos": 2,
    "tres": 3,
    "cuatro": 4,
    "cinco": 5,
    "seis": 6,
    "siete": 7,
    "ocho": 8,
    "nueve": 9,
    "diez": 10,
    "once": 11,
    "doce": 12,
}


def extract_room_info(text, nlp):
    """Extrae habitaciones, baños y lista de entidades ROOM."""
    doc = nlp(text)
    filters = {"habitaciones": [], "baños": [], "rooms_detected": []}
    for ent in doc.ents:
        if ent.label_ == "ROOM":
            filters["rooms_detected"].append(ent.text)

            lemmas = [token.lemma_.lower() for token in ent]
            text_norm = " ".join(lemmas)
            # Detectar número
            num = None
            for token in ent:
                token_text = token.text.lower().lstrip("-")

                try:
                    num = abs(int(token_text))
                    break
                except ValueError:
                    pass
                if token_text in NUMEROS_ES:
                    num = NUMEROS_ES[token_text]
                    break

            if num is None:
                num = 1

            if "habitación" in text_norm or "dormitorio" in text_norm:
                filters["habitaciones"].append(num)
            elif "baño" in text_norm:
                filters["baños"].append(num)

    # Tomar el máximo encontrado si hay múltiples menciones
    final_habs = max(filters["habitaciones"]) if filters["habitaciones"] else None
    final_baños = max(filters["baños"]) if filters["baños"] else None

    return final_habs, final_baños, filters["rooms_detected"]


def extract_room_info_from_doc(doc):
    filters = {"habitaciones": [], "baños": [], "rooms_detected": []}

    for ent in doc.ents:
        if ent.label_ == "ROOM":
            filters["rooms_detected"].append(ent.text)
            lemmas = [token.lemma_.lower() for token in ent]
            text_norm = " ".join(lemmas)

            num = None
            for i, token in enumerate(ent):
                token_text = token.text.lower().lstrip("-")
                try:
                    num = abs(int(token_text))
                    break
                except ValueError:
                    pass
                if token_text in NUMEROS_ES:
                    num = NUMEROS_ES[token_text]
                    break

            if num is None:
                num = 1

            if "habitación" in text_norm or "dormitorio" in text_norm:
                filters["habitaciones"].append(num)
            elif "baño" in text_norm:
                filters["baños"].append(num)

    # Drop duplicates
    filters["rooms_detected"] = list(dict.fromkeys(filters["rooms_detected"]))

    return {
        "habitaciones": max(filters["habitaciones"])
        if filters["habitaciones"]
        else None,
        "baños": max(filters["baños"]) if filters["baños"] else None,
        "rooms_detected": filters["rooms_detected"],
    }


def extract_other_labels(doc):
    data = {
        "area_detected": [],
        "caracteristicas_detected": [],
        "ubicacion_detected": [],
        "precio_detected": [],
    }

    for ent in doc.ents:
        if ent.label_ == "AREA":
            data["area_detected"].append(ent.text)
        elif ent.label_ == "CARACTERISTICAS":
            data["caracteristicas_detected"].append(ent.text)
        elif ent.label_ == "UBICACION":
            data["ubicacion_detected"].append(ent.text)
        elif ent.label_ in {"MONEY", "PRECIO"}:
            data["precio_detected"].append(ent.text)

    # Eliminar duplicados manteniendo orden
    for key in data:
        data[key] = list(dict.fromkeys(data[key]))

    return data


def filter_caracteristicas(doc, caracteristicas_validas):
    """Filtra CARACTERISTICAS según un conjunto de válidas, usando lemas."""
    filtered = []

    for ent in doc.ents:
        if ent.label_ != "CARACTERISTICAS":
            continue

        lemma_text = " ".join([token.lemma_.lower() for token in ent])
        for val in caracteristicas_validas:
            if val in lemma_text:
                filtered.append(val)
                break

    return list(dict.fromkeys(filtered))


def process_text_row(text, nlp, caracteristicas_validas=None):
    doc = nlp(str(text))
    info1 = extract_room_info_from_doc(doc)
    info2 = extract_other_labels(doc)
    caracteristicas_filtradas = (
        filter_caracteristicas(doc, caracteristicas_validas)
        if caracteristicas_validas
        else None
    )
    return {**info1, **info2, "caracteristicas": caracteristicas_filtradas}


def extract_price_number(precio_detected):
    if not precio_detected:
        return None

    for text in precio_detected:
        # Buscar secuencia numérica con punto o coma como separador
        matches = re.findall(r"[\d.,]+", text)
        for raw in matches:
            # Como no sé qué separador usa el usuario, asumo que es el separador de millares y lo elimino
            cleaned = raw.replace(".", "").replace(",", "")
            try:
                value = int(cleaned)
                # TODO: Tendria sentido aplicar un filtro de este tipo?
                # if 1000 < value < 10000000:  # Entre 1.000€ y 10M€
                #     return value
                return value
            except ValueError:
                continue
    return None


def extract_area_number(area_detected):
    if not area_detected:
        return None
    areas = []
    for text in area_detected:
        matches = re.findall(r"[\d.,]+", text)
        for raw in matches:
            cleaned = raw.replace(".", "").replace(",", "")
            try:
                value = int(cleaned)
                # Considerar áreas razonables
                if 20 <= value <= 2000:
                    areas.append(value)
            except ValueError:
                continue

    return max(areas) if areas else None
