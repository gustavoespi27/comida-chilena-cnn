"""Paso 3: aplica la revisión visual de las hojas de contacto y genera el dataset final.

- Las imágenes descartadas se MUEVEN (no se borran) a dataset_descartes/revision_manual/<clase>/
- Las imágenes con la clase equivocada se mueven a la carpeta correcta.
- El resultado queda en dataset_comida_chilena/ con su fuentes.csv actualizado.

Uso:  python scripts/aplicar_revision.py
"""
import csv
import shutil
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
CRUDO = BASE / "dataset_crudo"
FINAL = BASE / "dataset_comida_chilena"
DESCARTES = BASE / "dataset_descartes" / "revision_manual"

# Números de imagen descartados en la revisión visual (no son el plato terminado:
# ingredientes, preparación, textos/afiches, collages, personas, otros platos, etc.)
DESCARTAR = {
    "empanada": [26, 27, 46, 47, 48, 49, 50, 51, 52, 53, 55, 88, 95, 106, 137, 164, 191, 198, 200, 203, 205, 217],
    "completo": [15, 22, 28, 43, 57, 71, 72, 79, 80, 85, 137, 138, 145, 150, 151, 152, 153, 154, 155, 156, 157,
                 158, 159, 160, 161, 168, 169, 171, 172, 175, 178, 179, 180, 181, 182, 187, 188, 192, 193, 194,
                 196, 200, 201, 210, 215, 219],
    "chacarero": [12, 17, 18, 26, 40, 42, 64, 69, 75, 80, 86, 106, 107, 108, 109, 110, 111, 113, 116, 117, 120,
                  122, 125, 127, 129, 130, 132, 141, 145, 152, 160, 161, 162, 163, 170, 174, 176, 185, 188, 190,
                  193, 195, 196, 197, 198, 200, 203, 204, 205, 206],
    "sopaipilla": [27, 40, 44, 69, 71, 72, 74, 89, 90, 91, 120, 121, 153, 157, 158, 174, 175, 182, 183, 184, 188,
                   193, 194, 205, 206, 209, 210, 215, 218],
    "pastel_de_choclo": [16, 17, 18, 31, 32, 42, 86, 98, 99, 110, 122, 123, 137, 138, 166, 172, 174, 185, 190],
    "humitas": [15, 23, 24, 25, 26, 27, 28, 33, 41, 50, 51, 52, 54, 55, 57, 58, 59, 60, 79, 118, 119, 124, 136,
                139, 141, 142, 145, 159, 160, 163, 164, 165, 178, 180, 187, 188, 204, 208, 209, 210],
    "cazuela": [25, 29, 65, 69, 89, 93, 113, 129, 158, 161, 166, 167, 176, 177, 179, 212],
    "mote_con_huesillo": [31, 33, 40, 63, 64, 94, 103, 105, 108, 110, 111, 113, 114, 115, 117, 118, 119, 120,
                          124, 125, 126, 127, 128, 129, 132, 134, 138, 144, 154],
}

# Imágenes que estaban en la clase equivocada: (clase_origen, número) -> clase_correcta
RECLASIFICAR = {
    ("chacarero", 213): "completo",
    ("pastel_de_choclo", 8): "humitas",
}


def main():
    if FINAL.exists():
        raise SystemExit(f"{FINAL} ya existe; bórrelo si quiere regenerarlo.")
    shutil.copytree(CRUDO, FINAL)
    fuentes = {(r["clase"], r["archivo"]): r for r in csv.DictReader(open(CRUDO / "fuentes.csv", encoding="utf-8"))}

    for clase, numeros in DESCARTAR.items():
        destino = DESCARTES / clase
        destino.mkdir(parents=True, exist_ok=True)
        for n in numeros:
            archivo = FINAL / clase / f"{clase}_{n:04d}.jpg"
            if archivo.exists():
                shutil.move(str(archivo), destino / archivo.name)
            else:
                print("  aviso: no existe", archivo.name)

    for (origen, n), correcta in RECLASIFICAR.items():
        nombre = f"{origen}_{n:04d}.jpg"
        nuevo = f"{correcta}_reclasificada_{origen}_{n:04d}.jpg"
        shutil.move(str(FINAL / origen / nombre), FINAL / correcta / nuevo)
        fila = dict(fuentes.pop((origen, nombre)), clase=correcta, archivo=nuevo)
        fuentes[(correcta, nuevo)] = fila

    (FINAL / "fuentes.csv").unlink()
    filas = []
    print("Dataset final:")
    total = 0
    for clase_dir in sorted(d for d in FINAL.iterdir() if d.is_dir()):
        archivos = sorted(p.name for p in clase_dir.glob("*.jpg"))
        total += len(archivos)
        print(f"  {clase_dir.name:20s} {len(archivos)}")
        filas += [fuentes[(clase_dir.name, a)] for a in archivos]
    print(f"  {'TOTAL':20s} {total}")

    # fuentes.csv queda FUERA de la carpeta de clases para que no se cuente como imagen
    with open(BASE / "fuentes_dataset_comida_chilena.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["clase", "archivo", "url_imagen", "pagina_origen", "busqueda"])
        w.writeheader()
        w.writerows(filas)


if __name__ == "__main__":
    main()
