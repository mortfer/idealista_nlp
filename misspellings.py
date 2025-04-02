import symspellpy
from symspellpy import SymSpell, Verbosity

# Cargar diccionario en español
sym_spell = SymSpell()
sym_spell.load_dictionary(
    "es-100l.txt", 0, 1
)  # Download it from https://github.com/wolfgarbe/SymSpell/blob/master/SymSpell.FrequencyDictionary/es-100l.txt


def correct_text(text):
    corrected = sym_spell.lookup(text, Verbosity.CLOSEST, max_edit_distance=2)
    return corrected[0].term if corrected else text


# Prueba de corrección
print(correct_text("abitaciones"))  # -> "habitaciones"
print(correct_text("3 habitacions"))  # -> "3 habitacions" Mal!

import spacy
import contextualSpellCheck

nlp = spacy.load("es_core_news_lg")
contextualSpellCheck.add_to_pipe(nlp)
doc = nlp(
    "Me interesa un ático entre 60 y 90 metros cuadrados, 3 dormitorios, con un presupuesto máximo de 300 mil dólares"
)

print(doc._.performed_spellCheck)  # Fatal
print(doc._.outcome_spellCheck)
