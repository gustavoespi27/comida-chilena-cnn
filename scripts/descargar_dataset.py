"""Descarga imágenes de comida chilena y las ordena en una carpeta por clase.

El buscador entrega ~35 resultados por consulta, por eso cada clase usa muchas
consultas distintas y se eliminan las URLs repetidas.

Uso:  python scripts/descargar_dataset.py
Resultado: dataset_crudo/<clase>/*.jpg  +  dataset_crudo/fuentes.csv
"""
import csv
import hashlib
import io
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
from ddgs import DDGS
from PIL import Image

BASE = Path(__file__).resolve().parent.parent
DESTINO = BASE / "dataset_crudo"
OBJETIVO = 220   # candidatos máximos por clase (luego se limpia)
MIN_LADO = 150   # descarta miniaturas

CLASES = {
    "empanada": [
        "empanada de pino chilena", "empanadas de pino horno", "empanada chilena", "empanadas chilenas receta",
        "chilean empanada de pino", "empanadas fiestas patrias", "empanada de pino casera", "empanadas de horno chile",
        "empanada pino aceituna huevo", "empanadas chilenas dieciocho", "chilean baked empanadas", "empanada de pino cortada",
    ],
    "completo": [
        "completo italiano", "completo chileno", "completo italiano palta tomate mayo", "chilean completo hot dog",
        "completos chilenos", "completo dinamico", "completo italiano receta", "completo con palta",
        "completo as italiano", "completo chileno receta", "chilean hot dog italiano", "completos caseros chile",
    ],
    "sopaipilla": [
        "sopaipillas chilenas", "sopaipilla chilena", "sopaipillas con pebre", "sopaipillas pasadas",
        "sopaipillas de zapallo", "chilean sopaipillas", "sopaipillas receta chile", "sopaipilla con mostaza",
        "sopaipillas caseras", "sopaipillas fritas zapallo", "sopaipillas pasadas chancaca", "sopaipillas carrito chile",
    ],
    "chacarero": [
        "sandwich chacarero", "chacarero chileno", "churrasco chacarero", "chacarero sandwich chile",
        "chilean chacarero sandwich", "chacarero porotos verdes aji", "sandwich chacarero receta", "chacarero de lomo",
        "chacarero pan frica", "chacarero churrasco tomate", "chacarero fuente de soda", "sanguche chacarero",
    ],
    "pastel_de_choclo": [
        "pastel de choclo chileno", "pastel de choclo en greda", "pastel de choclo", "pastel de choclo receta",
        "chilean pastel de choclo", "pastel de choclo con pino", "pastel de choclo casero", "pastel de choclo gratinado",
        "pastel de choclo azucar", "pastel de choclo fuente", "chilean corn pie", "pastel de choclo plato",
    ],
    "humitas": [
        "humitas chilenas", "humitas en hoja de choclo", "humita chilena", "humitas receta chile",
        "chilean humitas", "humitas con ensalada chilena", "humitas caseras", "humitas cocidas hoja",
        "humitas plato", "humitas abiertas", "humitas de choclo", "humitas con tomate",
    ],
    "cazuela": [
        "cazuela de vacuno chilena", "cazuela de ave chilena", "cazuela chilena", "cazuela receta chile",
        "chilean cazuela soup", "cazuela de vacuno", "cazuela de pollo con zapallo choclo", "cazuela chilena plato",
        "cazuela de ave", "cazuela de pava", "cazuela casera chile", "cazuela con papas zapallo",
    ],
    "mote_con_huesillo": [
        "mote con huesillo", "mote con huesillos vaso", "mote con huesillo chileno", "mote con huesillo receta",
        "chilean mote con huesillo", "mote con huesillos", "mote huesillo verano", "mote con huesillo vaso plastico",
        "mote con huesillo carrito", "mote con huesillo casero", "huesillos con mote", "mote con huesillo jarra",
    ],
}

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124 Safari/537.36"}


def buscar(clase, consultas):
    urls, vistos = [], set()
    with DDGS() as ddgs:
        for consulta in consultas:
            try:
                resultados = ddgs.images(consulta, max_results=100)
            except Exception as e:
                print(f"  [{clase}] fallo la búsqueda '{consulta}': {e}")
                time.sleep(3)
                continue
            for r in resultados:
                url = r.get("image")
                if url and url not in vistos:
                    vistos.add(url)
                    urls.append((url, r.get("url", ""), consulta))
            time.sleep(1)
    return urls[:OBJETIVO * 2]


def descargar(item):
    url, pagina, consulta = item
    try:
        resp = requests.get(url, headers=HEADERS, timeout=10)
        resp.raise_for_status()
        img = Image.open(io.BytesIO(resp.content))
        img.load()
        if min(img.size) < MIN_LADO:
            return None
        return url, pagina, consulta, img.convert("RGB")
    except Exception:
        return None


def main():
    DESTINO.mkdir(exist_ok=True)
    registros = []
    for clase, consultas in CLASES.items():
        carpeta = DESTINO / clase
        carpeta.mkdir(exist_ok=True)
        candidatos = buscar(clase, consultas)
        with ThreadPoolExecutor(max_workers=12) as pool:
            descargadas = [r for r in pool.map(descargar, candidatos) if r]
        guardadas, hashes = 0, set()
        for url, pagina, consulta, img in descargadas:
            if guardadas >= OBJETIVO:
                break
            h = hashlib.md5(img.tobytes()).hexdigest()
            if h in hashes:  # copia exacta
                continue
            hashes.add(h)
            guardadas += 1
            nombre = f"{clase}_{guardadas:04d}.jpg"
            img.save(carpeta / nombre, "JPEG", quality=92)
            registros.append({"clase": clase, "archivo": nombre, "url_imagen": url,
                              "pagina_origen": pagina, "busqueda": consulta})
        print(f"{clase}: {len(candidatos)} URLs encontradas -> {guardadas} imágenes guardadas")

    with open(DESTINO / "fuentes.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["clase", "archivo", "url_imagen", "pagina_origen", "busqueda"])
        w.writeheader()
        w.writerows(registros)


if __name__ == "__main__":
    main()
