"""Paso 4: renumera las imágenes de cada clase de forma correlativa (clase_0001.jpg, clase_0002.jpg, ...)
y actualiza fuentes_dataset_comida_chilena.csv (agrega la columna archivo_anterior).

Uso:  python scripts/renumerar.py
"""
import csv
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
DATASET = BASE / "dataset_comida_chilena"
FUENTES = BASE / "fuentes_dataset_comida_chilena.csv"


def main():
    filas = {(r["clase"], r["archivo"]): r for r in csv.DictReader(open(FUENTES, encoding="utf-8"))}
    nuevas = []
    for clase_dir in sorted(d for d in DATASET.iterdir() if d.is_dir()):
        clase = clase_dir.name
        archivos = sorted(clase_dir.glob("*.jpg"))
        # 1) nombres temporales, para no pisar un archivo que ya tenga el nombre de destino
        temporales = []
        for i, ruta in enumerate(archivos, 1):
            tmp = ruta.with_name(f"__tmp_{i:04d}.jpg")
            ruta.rename(tmp)
            temporales.append((ruta.name, tmp))
        # 2) nombres definitivos
        for i, (anterior, tmp) in enumerate(temporales, 1):
            nuevo = f"{clase}_{i:04d}.jpg"
            tmp.rename(clase_dir / nuevo)
            fila = filas.get((clase, anterior), {"clase": clase, "url_imagen": "", "pagina_origen": "", "busqueda": ""})
            nuevas.append({**fila, "archivo": nuevo, "archivo_anterior": anterior})
        print(f"{clase:20s} {len(archivos)} -> {clase}_0001.jpg ... {clase}_{len(archivos):04d}.jpg")

    with open(FUENTES, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["clase", "archivo", "archivo_anterior", "url_imagen", "pagina_origen", "busqueda"])
        w.writeheader()
        w.writerows(nuevas)
    print("Total:", len(nuevas), "| fuentes actualizadas en", FUENTES.name)


if __name__ == "__main__":
    main()
