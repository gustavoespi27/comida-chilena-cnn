"""Genera NB03 y NB05 adaptados al dataset de comida chilena a partir de las guías originales.

Uso:  python scripts/generar_notebooks.py <carpeta_con_notebooks_originales>
Resultado (en la carpeta del proyecto):
  NB03_Preparacion_Dataset_Comida_Chilena.ipynb
  NB05_CNN_Comida_Chilena.ipynb
"""
import json
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
ORIG = Path(sys.argv[1])

NB03_IN = ORIG / "NB03_Preparacion_Gestion_Dataset_Imagenes.ipynb"
NB05_IN = ORIG / "NB05_Construccion_Entrenamiento_CNN.ipynb"
NB03_OUT = BASE / "NB03_Preparacion_Dataset_Comida_Chilena.ipynb"
NB05_OUT = BASE / "NB05_CNN_Comida_Chilena.ipynb"

CLASES = ["cazuela", "chacarero", "completo", "empanada", "humitas",
          "mote_con_huesillo", "pastel_de_choclo", "sopaipilla"]

ARBOL = "```text\ndataset_comida_chilena/\n" + "".join(
    f"{'└' if i == len(CLASES) - 1 else '├'}── {c}/\n" for i, c in enumerate(CLASES)) + "```"

REPO = "gustavoespi27/comida-chilena-cnn"

def badge(archivo):
    return (f"[![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)]"
            f"(https://colab.research.google.com/github/{REPO}/blob/main/{archivo})\n")

# Obtiene el dataset: si el notebook se ejecuta dentro del repositorio (PC local) usa esa carpeta;
# si no (Colab), clona el repositorio público de GitHub.
CELDA_REPO = f'''# Obtener el dataset desde GitHub
import subprocess
from pathlib import Path

REPO_URL = "https://github.com/{REPO}.git"

if Path("dataset_comida_chilena").is_dir():
    BASE_DIR = Path(".")                     # el notebook ya está dentro del repositorio
else:
    BASE_DIR = Path("comida-chilena-cnn")
    if not BASE_DIR.is_dir():                # clonar solo la primera vez
        subprocess.run(["git", "clone", "--depth", "1", REPO_URL, str(BASE_DIR)], check=True)

DATASET_DIR = BASE_DIR / "dataset_comida_chilena"
if not DATASET_DIR.is_dir():
    raise FileNotFoundError(f"No se encontró el dataset: {{DATASET_DIR}}")
print("Dataset:", DATASET_DIR.resolve())
'''

# NB05: Drive se usa solo para conservar los modelos entrenados (Colab borra sus archivos al cerrar la sesión).
CELDA_DRIVE_MODELOS = '''# Conectar Google Drive para GUARDAR los modelos entrenados.
# Los archivos creados en Colab se borran al cerrar la sesión; en Drive se conservan.
# Fuera de Colab, los modelos se guardan en la carpeta local "modelos".
import os

try:
    from google.colab import drive
    drive.mount('/content/drive')
    CARPETA_MODELOS = "/content/drive/MyDrive/MachineLearning2026"
except ImportError:
    CARPETA_MODELOS = "modelos"

os.makedirs(CARPETA_MODELOS, exist_ok=True)
print("Los modelos se guardarán en:", CARPETA_MODELOS)
'''


# ---------------------------------------------------------------- utilidades
def cargar(ruta):
    nb = json.load(open(ruta, encoding="utf-8"))
    for c in nb["cells"]:
        c["source"] = "".join(c["source"])
    return nb


def guardar(nb, ruta):
    for i, c in enumerate(nb["cells"]):
        c["id"] = f"celda-{i:02d}"
        c["source"] = c["source"].splitlines(keepends=True)
        if c["cell_type"] == "code":
            c["outputs"], c["execution_count"] = [], None
        c.get("metadata", {}).pop("id", None)
    with open(ruta, "w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
        f.write("\n")


def md(texto):
    return {"cell_type": "markdown", "metadata": {}, "source": texto}


def code(texto):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": texto}


def celda(nb, inicio):
    """Devuelve la celda cuyo texto empieza con `inicio` (falla si no hay exactamente una)."""
    encontradas = [c for c in nb["cells"] if c["source"].lstrip().startswith(inicio)]
    assert len(encontradas) == 1, f"Se esperaba 1 celda que empiece con {inicio!r}, hay {len(encontradas)}"
    return encontradas[0]


def reemplazar(c, viejo, nuevo):
    assert viejo in c["source"], f"No se encontró en la celda: {viejo!r}"
    c["source"] = c["source"].replace(viejo, nuevo)


def insertar_despues(nb, c, nuevas):
    i = nb["cells"].index(c)
    nb["cells"][i + 1:i + 1] = nuevas


# ---------------------------------------------------------------- NB03
def generar_nb03():
    nb = cargar(NB03_IN)

    c = nb["cells"][0]
    c["source"] = (
        badge(NB03_OUT.name) +
        "# NB03 — Preparación y gestión del dataset de imágenes: comida chilena\n"
        "**Dataset del proyecto:** fotografías de 8 platos típicos chilenos (`" + "`, `".join(CLASES) + "`).\n\n"
        "Adaptación de la guía *NB03 — Preparación y gestión del dataset de imágenes* (Taller Fundamentos "
        "Data Science Python, Machine Learning 2026), que originalmente usa TensorFlow Flower Photos. "
        "Se aplica el mismo procedimiento al dataset propio.\n"
    )

    c = celda(nb, "## Introducción")
    reemplazar(c, "Usaremos el dataset público **TensorFlow Flower Photos**, organizado en cinco categorías: "
                  "`daisy`, `dandelion`, `roses`, `sunflowers` y `tulips`. El objetivo es que cada estudiante "
                  "comprenda un procedimiento transferible a su propio proyecto de clasificación de imágenes.",
               "Usaremos un **dataset propio de comida chilena**, organizado en ocho categorías: `"
               + "`, `".join(CLASES) + "`. Las imágenes se obtuvieron mediante búsquedas web, se filtraron "
               "automáticamente (duplicados y tamaño mínimo) y se revisaron manualmente para eliminar "
               "fotografías que no correspondían a cada plato.")

    c = celda(nb, "## 2. Fuente de datos")
    c["source"] = (
        "## 2. Fuente de datos: repositorio de GitHub\n\n"
        "**Objetivo.** Obtener el dataset del proyecto. Las imágenes de comida chilena están publicadas en el "
        f"repositorio [`{REPO}`](https://github.com/{REPO}), junto con este notebook. **No se requiere Google "
        "Drive.**\n\n"
        "**Qué hace el código:** si el notebook se ejecuta dentro del repositorio (por ejemplo, en un PC donde "
        "se clonó), usa esa carpeta; en Colab, descarga el repositorio con `git clone` (solo la primera vez). "
        "Los archivos descargados en Colab son temporales y desaparecen al terminar la sesión.\n\n"
        "**Qué verificar:** la celda debe mostrar la ruta de `dataset_comida_chilena`.\n"
    )
    celda(nb, "# OPCIONAL: descomente")["source"] = CELDA_REPO

    c = celda(nb, "## 3. Descargar y ubicar el dataset")
    c["source"] = (
        "## 3. Ubicar el dataset\n\n"
        "**Objetivo.** Comprobar el conjunto de imágenes etiquetadas mediante su estructura de "
        "carpetas. Cada subcarpeta representa una clase; por ejemplo, `empanada` contiene fotografías "
        "etiquetadas como empanadas.\n\n"
        "**Qué hace el código:** lista las carpetas de clase que contiene `DATASET_DIR`.\n\n"
        "**Qué verificar:** `DATASET_DIR` debe contener las ocho carpetas indicadas. "
        "**Pregunta:** ¿qué ocurriría si mezcláramos imágenes de dos clases en la misma carpeta?\n\n"
        + ARBOL + "\n\n"
        "El archivo con el origen de cada imagen (`fuentes_dataset_comida_chilena.csv`) se guarda **fuera** "
        "de esta carpeta, para que no se cuente como imagen.\n"
    )
    celda(nb, "URL = ")["source"] = (
        'carpetas_clase = sorted(p.name for p in DATASET_DIR.iterdir() if p.is_dir())\n'
        'print("Carpetas de clase:", len(carpetas_clase))\n'
        'for nombre in carpetas_clase:\n'
        '    print(" -", nombre)\n'
    )

    reemplazar(celda(nb, "## 5. Analizar el balance"), "reconoce bien todas las flores", "reconoce bien todos los platos")
    reemplazar(celda(nb, "## 13. Data Augmentation"), "que la flor siga siendo reconocible", "que el plato siga siendo reconocible")
    reemplazar(celda(nb, "## 15. Verificar etiquetas"), "`roses` y `tulips`", "`humitas` y `pastel_de_choclo`")

    celda(nb, "ficha_dataset = {")["source"] = '''ficha_dataset = {
    "fuente": "Imágenes web de comida chilena (búsquedas en buscadores de imágenes)",
    "procedencia": "Ver fuentes_dataset_comida_chilena.csv (URL de la imagen y página de origen de cada archivo)",
    "limpieza": "Eliminación automática de duplicados y miniaturas (<150 px) + revisión manual de todas las imágenes",
    "permisos_de_uso": "Uso académico, sin redistribución: las imágenes pertenecen a sus autores",
    "numero_imagenes_validas": len(archivos_validos),
    "numero_clases": len(class_names),
    "clases": class_names,
    "distribucion_por_clase": df_resumen.set_index("Clase")["Cantidad"].to_dict(),
    "tamano_imagen": IMG_SIZE,
    "batch_size": BATCH_SIZE,
    "particion": {"train": len(X_train), "validation": len(X_val), "test": len(X_test)},
    "normalizacion": "0-1",
    "data_augmentation_solo_train": ["RandomFlip horizontal", "RandomRotation 0.10", "RandomZoom 0.10"],
    "archivos_problematicos": len(archivos_problematicos)
}
ficha_dataset
'''

    c = celda(nb, "## 17. Actividad")
    reemplazar(c, "Aplique el notebook al dataset seleccionado para el proyecto integrador y registre:",
               "Se aplicó el notebook al dataset de comida chilena. Se registra:")
    insertar_despues(nb, c, [
        md("### 17.1 Evidencia calculada\n\nLa siguiente celda reúne en una tabla los datos de los puntos 1, 2, "
           "3, 5, 6, 8 y 9, a partir de las salidas anteriores."),
        code('''resumen_por_clase = (
    df_imagenes.groupby("clase")
    .agg(imagenes=("archivo", "count"),
         ancho_min=("ancho", "min"), ancho_mediana=("ancho", "median"), ancho_max=("ancho", "max"),
         alto_min=("alto", "min"), alto_mediana=("alto", "median"), alto_max=("alto", "max"))
)
resumen_por_clase["porcentaje"] = (100 * resumen_por_clase["imagenes"] / resumen_por_clase["imagenes"].sum()).round(1)
resumen_por_clase["train"] = pd.Series(y_train).map(dict(enumerate(class_names))).value_counts()
resumen_por_clase["validation"] = pd.Series(y_val).map(dict(enumerate(class_names))).value_counts()
resumen_por_clase["test"] = pd.Series(y_test).map(dict(enumerate(class_names))).value_counts()

mayor, menor = resumen_por_clase["imagenes"].max(), resumen_por_clase["imagenes"].min()
print("1. Total de imágenes:", len(df_imagenes))
print("2. Número de clases:", len(class_names))
print("6. Archivos problemáticos:", len(archivos_problematicos))
print(f"8. Partición: train={len(X_train)}, validation={len(X_val)}, test={len(X_test)}")
print(f"9. Razón de desbalance (clase mayor / clase menor): {mayor / menor:.2f}")
assert not (set(X_train) & set(X_val) or set(X_train) & set(X_test) or set(X_val) & set(X_test)), "Hay rutas repetidas entre subconjuntos"
print("   Ninguna ruta se repite entre train, validation y test.")
resumen_por_clase
'''),
        md(RESPUESTAS_NB03),
    ])
    guardar(nb, NB03_OUT)


RESPUESTAS_NB03 = "### 17.2 Respuestas\n\n__RESPUESTAS_NB03__\n"


# ---------------------------------------------------------------- NB05
def generar_nb05():
    nb = cargar(NB05_IN)

    c = nb["cells"][0]
    reemplazar(c, c["source"].split("\n## Propósito")[0],
               badge(NB05_OUT.name) +
               "# NB05 — Construcción y entrenamiento de una CNN: comida chilena\n\n"
               "**Dataset del proyecto:** fotografías de 8 platos típicos chilenos (`" + "`, `".join(CLASES) + "`), "
               "preparado y documentado en el NB03.\n\n"
               "Adaptación de la guía *NB05 — Construcción y entrenamiento de una CNN con un dataset propio* "
               "(Taller Fundamentos Data Science Python, Machine Learning 2026).\n")
    reemplazar(c, "Este notebook utiliza Google Drive para acceder a las carpetas del proyecto; cada grupo debe "
                  "adaptar la ruta a su propia organización.",
               f"Este notebook descarga el dataset desde el repositorio de GitHub [`{REPO}`](https://github.com/{REPO}) "
               "y utiliza Google Drive solo para guardar los modelos entrenados.")

    reemplazar(celda(nb, "## 2. Conectar Google Drive"), "## 2. Conectar Google Drive",
               "## 2. Conectar Google Drive (para guardar los modelos)\n\n"
               "Los archivos creados en Colab se borran al cerrar la sesión. Por eso los modelos entrenados se "
               "guardan en `MyDrive/MachineLearning2026`. Autorice el acceso a Drive cuando Colab lo solicite.")
    celda(nb, "from google.colab import drive")["source"] = CELDA_DRIVE_MODELOS

    c = celda(nb, "## 3. Definir el dataset propio")
    reemplazar(c, c["source"][c["source"].index("Ejemplo ilustrativo"):c["source"].index("Compruebe que")],
               "Estructura del dataset de comida chilena:\n\n" + ARBOL + "\n\n")
    reemplazar(c, "**La ruta `DATASET_DIR` de la siguiente celda es un ejemplo:** sustitúyala por la ruta real de su carpeta en Google Drive.",
               "La siguiente celda obtiene el dataset: si el notebook se ejecuta dentro del repositorio usa esa "
               "carpeta; en Colab descarga el repositorio con `git clone` (solo la primera vez).")

    reemplazar(celda(nb, 'DATASET_DIR = "/content/drive'),
               'DATASET_DIR = "/content/drive/MyDrive/MachineLearning2026/dataset"',
               CELDA_REPO + 'DATASET_DIR = str(DATASET_DIR)  # image_dataset_from_directory recibe la ruta como texto')

    reemplazar(celda(nb, 'MODEL_PATH = '),
               'MODEL_PATH = "/content/drive/MyDrive/MachineLearning2026/cnn_inicial.keras"',
               'MODEL_PATH = f"{CARPETA_MODELOS}/cnn_comida_chilena_inicial_v1.keras"')

    c = celda(nb, "## 23. Actividad")
    reemplazar(c, "A partir del dataset del proyecto integrador:", "A partir del dataset de comida chilena:")
    insertar_despues(nb, c, [
        md("### 23.1 Capas convolucionales, pooling y parámetros (puntos 1 a 3)"),
        code('''def describir_capas(modelo):
    filas = []
    for capa in modelo.layers:
        tipo = type(capa).__name__
        if tipo in ("Conv2D", "MaxPooling2D", "GlobalAveragePooling2D", "Dense", "Dropout"):
            filas.append({
                "capa": capa.name,
                "tipo": tipo,
                "salida": str(capa.output.shape),
                "filtros/neuronas": getattr(capa, "filters", getattr(capa, "units", "")),
                "parámetros": capa.count_params(),
            })
    return pd.DataFrame(filas)

print("CNN inicial — parámetros totales:", model.count_params())
display(describir_capas(model))
print("CNN experimental — parámetros totales:", modelo_experimental.count_params())
display(describir_capas(modelo_experimental))
'''),
        md("### 23.2 Señales de overfitting o underfitting (puntos 4 y 5)\n\n"
           "Se comparan las curvas de ambos modelos y se resume la última época y la mejor época de validación."),
        code('''def diagnostico(historial, nombre):
    h = pd.DataFrame(historial.history)
    ultima = h.iloc[-1]
    mejor_epoca = int(h["val_loss"].idxmin()) + 1
    brecha = ultima["accuracy"] - ultima["val_accuracy"]
    if ultima["accuracy"] < 0.6 and ultima["val_accuracy"] < 0.6:
        lectura = "posible underfitting (accuracy baja en ambos conjuntos)"
    elif brecha > 0.10 or mejor_epoca < len(h) - 3:
        lectura = "posible overfitting (train supera a validación o val_loss deja de bajar)"
    else:
        lectura = "sin señales fuertes de overfitting ni underfitting"
    return {"modelo": nombre, "acc_train_final": round(ultima["accuracy"], 4),
            "acc_val_final": round(ultima["val_accuracy"], 4), "brecha_train_val": round(brecha, 4),
            "mejor_epoca_val_loss": mejor_epoca, "epocas": len(h), "lectura": lectura}

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
for historial, nombre, estilo in [(history, "inicial", "-"), (history_exp, "experimental", "--")]:
    h = pd.DataFrame(historial.history)
    axes[0].plot(h["accuracy"], estilo, label=f"{nombre} train")
    axes[0].plot(h["val_accuracy"], estilo, label=f"{nombre} validación")
    axes[1].plot(h["loss"], estilo, label=f"{nombre} train")
    axes[1].plot(h["val_loss"], estilo, label=f"{nombre} validación")
axes[0].set_title("Accuracy"); axes[1].set_title("Loss")
for ax in axes:
    ax.set_xlabel("Época"); ax.legend()
plt.tight_layout()
plt.show()

pd.DataFrame([diagnostico(history, "CNN inicial"), diagnostico(history_exp, "CNN experimental")])
'''),
        md("### 23.3 Comportamiento por clase de ambos modelos (punto 6)"),
        code('''from sklearn.metrics import precision_recall_fscore_support

def predecir(modelo):
    reales, predichas = [], []
    for images, labels in val_ds:
        reales.extend(labels.numpy())
        predichas.extend(np.argmax(modelo.predict(images, verbose=0), axis=1))
    return np.array(reales), np.array(predichas)

y_true_ini, y_pred_ini = predecir(model)
y_true_exp, y_pred_exp = predecir(modelo_experimental)

def f1_por_clase(y_t, y_p):
    _, _, f1, soporte = precision_recall_fscore_support(y_t, y_p, labels=range(NUM_CLASSES), zero_division=0)
    return f1, soporte

f1_ini, soporte = f1_por_clase(y_true_ini, y_pred_ini)
f1_exp, _ = f1_por_clase(y_true_exp, y_pred_exp)
por_clase = pd.DataFrame({"clase": class_names, "imagenes_validacion": soporte,
                          "f1_inicial": f1_ini.round(3), "f1_experimental": f1_exp.round(3)})
display(por_clase.sort_values("f1_inicial"))

# Pares de clases que más se confunden (modelo inicial)
cm_ini = confusion_matrix(y_true_ini, y_pred_ini, labels=range(NUM_CLASSES))
confusiones = [(class_names[i], class_names[j], int(cm_ini[i, j]))
               for i in range(NUM_CLASSES) for j in range(NUM_CLASSES) if i != j and cm_ini[i, j] > 0]
pd.DataFrame(sorted(confusiones, key=lambda x: -x[2])[:6], columns=["clase_real", "predicha_como", "cantidad"])
'''),
        md("### 23.4 Comparación y elección del modelo (puntos 7 a 10)\n\n"
           "Se elige el modelo con mayor *accuracy* de validación; en caso de empate se prefiere el que tiene "
           "menos parámetros. Se guarda con un nombre versionado y una ficha JSON con la información necesaria "
           "para reproducirlo."),
        code('''import json
from datetime import date

comparacion["F1_macro"] = [f1_ini.mean().round(4), f1_exp.mean().round(4)]
display(comparacion)

candidatos = [(val_accuracy, -model.count_params(), "CNN inicial", model),
              (exp_accuracy, -modelo_experimental.count_params(), "CNN experimental", modelo_experimental)]
acc_elegido, _, nombre_elegido, modelo_elegido = max(candidatos, key=lambda x: (round(x[0], 4), x[1]))
print(f"Modelo elegido: {nombre_elegido} (validation accuracy = {acc_elegido:.4f})")

VERSION = "v1"
RUTA_ELEGIDO = f"{CARPETA_MODELOS}/cnn_comida_chilena_elegido_{VERSION}.keras"
modelo_elegido.save(RUTA_ELEGIDO)

ficha_modelo = {
    "version": VERSION,
    "fecha": str(date.today()),
    "modelo": nombre_elegido,
    "archivo": RUTA_ELEGIDO,
    "clases_en_orden": class_names,
    "tamano_entrada": list(IMG_SIZE),
    "parametros": int(modelo_elegido.count_params()),
    "epocas": EPOCHS,
    "batch_size": BATCH_SIZE,
    "optimizador": "Adam (learning_rate=0.001)",
    "division": "80 % train / 20 % validation, seed 42",
    "validation_accuracy": round(float(acc_elegido), 4),
}
with open(f"{CARPETA_MODELOS}/cnn_comida_chilena_elegido_{VERSION}.json", "w", encoding="utf-8") as f:
    json.dump(ficha_modelo, f, ensure_ascii=False, indent=2)
ficha_modelo
'''),
        md(RESPUESTAS_NB05),
    ])
    guardar(nb, NB05_OUT)


RESPUESTAS_NB05 = "### 23.5 Respuestas\n\n__RESPUESTAS_NB05__\n"

if __name__ == "__main__":
    respuestas = BASE / "scripts" / "respuestas"
    for nombre in ("NB03", "NB05"):
        archivo = respuestas / f"{nombre}.md"
        texto = archivo.read_text(encoding="utf-8") if archivo.exists() else "*(pendiente)*"
        globals()[f"RESPUESTAS_{nombre}"] = globals()[f"RESPUESTAS_{nombre}"].replace(f"__RESPUESTAS_{nombre}__", texto)
    generar_nb03()
    generar_nb05()
    print("Generados:", NB03_OUT.name, NB05_OUT.name)
