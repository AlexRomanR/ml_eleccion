# app/services/kanban.py
import re
import cv2
import numpy as np


_PIPELINE = None

DATE_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
EMAIL_RE = re.compile(r"\b\S+@\S+\.\S+\b", re.IGNORECASE)

PRIORITY_MAP = {
    "BAJA": "BAJA",
    "MEDIA": "MEDIA",
    "ALTA": "ALTA",
}

def _get_pipeline():
    global _PIPELINE
    if _PIPELINE is None:
        import keras_ocr  # ✅ lazy import (NO se carga en el arranque)
        _PIPELINE = keras_ocr.pipeline.Pipeline()
    return _PIPELINE


def _box_center(box):
    # box: (4,2)
    x = float(np.mean(box[:, 0]))
    y = float(np.mean(box[:, 1]))
    return x, y


def _sort_reading_order(preds):
    """
    preds: list[(text, box)]
    Devuelve lista ordenada por y y luego x (lectura aproximada).
    """
    items = []
    for text, box in preds:
        cx, cy = _box_center(box)
        items.append((cy, cx, text.strip()))
    items.sort(key=lambda t: (t[0], t[1]))
    return [t[2] for t in items if t[2]]


def _clean_header_text(words):
    # filtramos numeritos y cosas cortas
    candidates = []
    for w in words:
        w2 = w.strip()
        if not w2:
            continue
        if w2.isdigit():
            continue
        if len(w2) <= 1:
            continue
        # evita "Agregar" en header si apareciera raro
        if "agregar" in w2.lower():
            continue
        candidates.append(w2)

    # en tu UI el header es una sola palabra o 2 ("Por hacer")
    # nos quedamos con lo más largo, o unimos 2 primeras si hace sentido
    if not candidates:
        return "Columna"

    # si hay 2 palabras con y parecida, keras_ocr puede separarlo: "Por" "hacer"
    # unimos si hay exactamente 2 y ninguna es muy larga
    if len(candidates) >= 2:
        joined = " ".join(candidates[:2])
        # heurística: si parece "Por hacer"
        if len(joined) <= 20:
            # preferir join si no es basura
            if any(c.isalpha() for c in joined):
                return joined

    return max(candidates, key=len)


def _find_columns_by_white_body(img_bgr):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    # blanco: baja saturación, alto valor
    mask = cv2.inRange(hsv, (0, 0, 180), (179, 45, 255))

    H, W = mask.shape
    mask[: int(H * 0.12), :] = 0  # quitamos cabecera

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8), iterations=2)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    rects = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        area = w * h
        if area > 50000 and h > int(H * 0.35) and w > int(W * 0.12):
            rects.append((x, y, w, h, area))

    rects = sorted(rects, key=lambda r: r[0])  # por X (izq→der)

    # ✅ YA NO recortes a 3: retorna todas
    return [(x, y, w, h) for (x, y, w, h, _) in rects]


def _find_cards_in_column(col_bgr):
    """
    Busca rectángulos internos (cards) dentro del cuerpo de una columna.
    Usa threshold adaptativo para capturar bordes suaves.
    """
    gray = cv2.cvtColor(col_bgr, cv2.COLOR_BGR2GRAY)

    th = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 5
    )
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=2)

    H, W = gray.shape[:2]
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cards = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        area = w * h

        # filtrar: muy chico / muy grande (la columna entera)
        if area < 12000:
            continue
        if w > int(W * 0.95) and h > int(H * 0.95):
            continue

        # card típica: bastante ancha y alta
        if w > int(W * 0.65) and h > int(H * 0.20):
            cards.append((x, y, w, h, area))

    # ordenar por Y (de arriba hacia abajo)
    cards.sort(key=lambda r: r[1])
    return [(x, y, w, h) for (x, y, w, h, _) in cards]

DATE_FUZZY = re.compile(r"\b(20\d{2})\s*[-/–]\s*(\d{2})\s*[-/–]\s*(\d{2})\b")
PTS_IN_TEXT = re.compile(r"(?<!#)\b(\d{1,3})\b\s*(?:pts|puntos?)\b", re.IGNORECASE)
NUM_ANY = re.compile(r"(?<!#)\b(\d{1,3})\b")

def _parse_task_from_words(words):
    cleaned = []
    for w in words:
        w2 = w.strip()
        if not w2:
            continue
        if EMAIL_RE.search(w2):
            continue
        if "agregar" in w2.lower() and "tarea" in w2.lower():
            continue
        cleaned.append(w2)

    joined = " ".join(cleaned)

    # ---- fecha (YYYY-MM-DD) robusta ----
    fecha = None
    m = DATE_FUZZY.search(joined)
    if m:
        fecha = f"{m.group(1)}-{m.group(2)}-{m.group(3)}"

    # ---- prioridad ----
    prioridad = None
    for w in cleaned:
        up = w.upper()
        if up in PRIORITY_MAP:
            prioridad = PRIORITY_MAP[up]
            break

    # ---- puntos: 1) si aparece "X puntos" en el texto ----
    puntos = None
    m = PTS_IN_TEXT.search(joined)
    if m:
        n = int(m.group(1))
        if 0 <= n <= 500:
            puntos = n

    # ---- puntos: 2) fallback buscando cualquier numerito suelto (evita años grandes) ----
    if puntos is None:
        # buscamos candidatos, evitando tokens que parecen fecha completa
        candidates = []
        for w in cleaned:
            if DATE_FUZZY.search(w):
                continue
            for s in NUM_ANY.findall(w):
                n = int(s)
                if 0 <= n <= 500 and n < 1900:   # evita 2025
                    candidates.append(n)
        if candidates:
            # normalmente el punto es el número “chico” (5,6,8,50, etc.)
            puntos = min(candidates)

    # ---- título/desc ----
    def is_noise(x):
        if DATE_FUZZY.search(x): return True
        if x.upper() in PRIORITY_MAP: return True
        # no filtres números aquí, porque el título puede tener "#7" y no importa
        return False

    text_lines = [w for w in cleaned if not is_noise(w)]
    titulo = text_lines[0] if text_lines else "Tarea"
    descripcion = text_lines[1] if len(text_lines) >= 2 else None

    return {
        "titulo": titulo,
        "descripcion": descripcion,
        "prioridad": prioridad,
        "puntos": puntos,
        "fechaLimite": fecha,
    }


class KanbanExtractor:
    def extract_from_bytes(self, image_bytes):
        # decode bytes -> cv2 image
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return {"error": "Could not decode image"}

        cols = _find_columns_by_white_body(img)
        if len(cols) < 2:
            return {"error": "No se detectaron columnas suficientes"}

        pipe = _get_pipeline()

        out_cols = []
        for (x, y, w, h) in cols:
            # header: zona encima del cuerpo
            header_y0 = 0
            header_y1 = max(0, y)  # hasta donde empieza el blanco
            header = img[header_y0:header_y1, x:x + w]

            # cuerpo de columna (blanco)
            body = img[y:y + h, x:x + w]

            # OCR header
            header_rgb = cv2.cvtColor(header, cv2.COLOR_BGR2RGB)
            header_preds = pipe.recognize([header_rgb])[0] if header_rgb.size else []
            header_words = _sort_reading_order(header_preds)
            col_name = _clean_header_text(header_words)

            # detectar cards
            cards = _find_cards_in_column(body)

            tasks = []
            if cards:
                card_imgs = []
                for (cx, cy, cw, ch) in cards:
                    card = body[cy:cy + ch, cx:cx + cw]
                    card_imgs.append(cv2.cvtColor(card, cv2.COLOR_BGR2RGB))

                pred_groups = pipe.recognize(card_imgs)

                for preds in pred_groups:
                    words = _sort_reading_order(preds)

                    # si la card es el botón "Agregar tarea", la ignoramos
                    joined = " ".join(w.lower() for w in words)
                    if "agregar" in joined and "tarea" in joined:
                        continue

                    tasks.append(_parse_task_from_words(words))

            out_cols.append({
                "nombre": col_name,
                "orden": len(out_cols) + 1,
                "tasks": tasks
            })

        # shape compatible con tu mapper del front
        return {
            "project": {
                "nombre": "Proyecto (desde imagen)",
                "descripcion": None,
                "estado": "Activo",
                "fechaInicio": None,
                "fechaFin": None
            },
            "columns": out_cols
        }
