"""Paso 2: elimina casi-duplicados y genera hojas de contacto para revisión visual.

Uso:
  python scripts/limpiar_y_revisar.py duplicados      -> mueve casi-duplicados a dataset_descartes/
  python scripts/limpiar_y_revisar.py hojas           -> crea revision/<clase>_N.jpg con miniaturas numeradas
"""
import shutil
import sys
from pathlib import Path

import imagehash
from PIL import Image, ImageDraw, ImageFont

BASE = Path(__file__).resolve().parent.parent
DATASET = BASE / "dataset_crudo"
DESCARTES = BASE / "dataset_descartes"
REVISION = BASE / "revision"
UMBRAL = 6  # distancia de Hamming del hash perceptual: <= 6 se considera la misma foto


def imagenes(clase_dir):
    return sorted(p for p in clase_dir.iterdir() if p.suffix.lower() == ".jpg")


def duplicados():
    vistos = []  # (hash, ruta) de todas las clases, para detectar también fotos repetidas entre clases
    movidas = 0
    for clase_dir in sorted(d for d in DATASET.iterdir() if d.is_dir()):
        for ruta in imagenes(clase_dir):
            h = imagehash.phash(Image.open(ruta))
            original = next((r for hv, r in vistos if h - hv <= UMBRAL), None)
            if original:
                destino = DESCARTES / "duplicados" / clase_dir.name
                destino.mkdir(parents=True, exist_ok=True)
                shutil.move(str(ruta), destino / ruta.name)
                movidas += 1
                if original.parent != ruta.parent:
                    print(f"  repetida entre clases: {ruta.name} = {original.parent.name}/{original.name}")
            else:
                vistos.append((h, ruta))
    print("Casi-duplicados movidos:", movidas)
    for clase_dir in sorted(d for d in DATASET.iterdir() if d.is_dir()):
        print(f"  {clase_dir.name}: {len(imagenes(clase_dir))}")


def hojas(columnas=10, filas=6, lado=150):
    REVISION.mkdir(exist_ok=True)
    fuente = ImageFont.load_default(size=18)
    for clase_dir in sorted(d for d in DATASET.iterdir() if d.is_dir()):
        rutas = imagenes(clase_dir)
        por_hoja = columnas * filas
        for n in range(0, len(rutas), por_hoja):
            hoja = Image.new("RGB", (columnas * lado, filas * lado), "white")
            dibujo = ImageDraw.Draw(hoja)
            for i, ruta in enumerate(rutas[n:n + por_hoja]):
                img = Image.open(ruta).convert("RGB")
                img.thumbnail((lado - 4, lado - 4))
                x, y = (i % columnas) * lado, (i // columnas) * lado
                hoja.paste(img, (x + 2, y + 2))
                numero = ruta.stem.split("_")[-1].lstrip("0")
                dibujo.rectangle([x + 2, y + 2, x + 48, y + 24], fill="yellow")
                dibujo.text((x + 5, y + 3), numero, fill="black", font=fuente)
            hoja.save(REVISION / f"{clase_dir.name}_{n // por_hoja + 1}.jpg", quality=85)
    print("Hojas creadas en", REVISION)


if __name__ == "__main__":
    {"duplicados": duplicados, "hojas": hojas}[sys.argv[1]]()
