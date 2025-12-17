import pandas as pd
import numpy as np
from app.db import get_engine
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sqlalchemy import text

SQL_FEATURES = """
WITH ultima_col AS (
  SELECT c.proyecto_id, MAX(c.orden) AS max_orden
  FROM columna_kanban c
  GROUP BY c.proyecto_id
),
last_move AS (
  SELECT h.tarea_id, MAX(h.cambiado_en) AS last_change
  FROM historial_estado_tarea h
  GROUP BY h.tarea_id
),
dias_sin AS (
  SELECT t.id AS tarea_id,
         TIMESTAMPDIFF(DAY, lm.last_change, NOW()) AS dias_sin_mov
  FROM last_move lm
  JOIN tarea t ON t.id = lm.tarea_id
  WHERE t.proyecto_id = :proyectoId
),
back_moves AS (
  SELECT h.tarea_id,
         SUM(CASE WHEN c_from.orden > c_to.orden THEN 1 ELSE 0 END) AS back_moves
  FROM historial_estado_tarea h
  JOIN columna_kanban c_from ON c_from.id = h.de_columna_id
  JOIN columna_kanban c_to   ON c_to.id   = h.a_columna_id
  GROUP BY h.tarea_id
),
reopen_from_done AS (
  SELECT h.tarea_id,
         SUM(CASE
               WHEN c_from.orden = uc.max_orden AND c_to.orden < uc.max_orden
               THEN 1 ELSE 0 END) AS reopen_expl
  FROM historial_estado_tarea h
  JOIN columna_kanban c_from ON c_from.id = h.de_columna_id
  JOIN columna_kanban c_to   ON c_to.id   = h.a_columna_id
  JOIN ultima_col uc ON uc.proyecto_id = c_from.proyecto_id
  GROUP BY h.tarea_id
),
fecha_inicio AS (
  SELECT tarea_id, MIN(cambiado_en) AS fecha_inicio
  FROM historial_estado_tarea
  GROUP BY tarea_id
),
fecha_fin AS (
  SELECT h.tarea_id, MIN(h.cambiado_en) AS fecha_fin
  FROM historial_estado_tarea h
  JOIN columna_kanban c ON c.id = h.a_columna_id
  JOIN ultima_col uc ON uc.proyecto_id = c.proyecto_id AND uc.max_orden = c.orden
  GROUP BY h.tarea_id
),
cycle AS (
  SELECT t.id AS tarea_id,
         CASE
           WHEN fi.fecha_fin IS NOT NULL AND si.fecha_inicio IS NOT NULL
           THEN TIMESTAMPDIFF(DAY, si.fecha_inicio, fi.fecha_fin)
           ELSE NULL
         END AS cycle_days
  FROM tarea t
  LEFT JOIN fecha_inicio si ON si.tarea_id = t.id
  LEFT JOIN fecha_fin fi    ON fi.tarea_id = t.id
  WHERE t.proyecto_id = :proyectoId
)
SELECT
  t.id AS tarea_id,
  COALESCE(d.dias_sin_mov, 0) AS dias_sin_mov,
  COALESCE(b.back_moves, 0)   AS back_moves,
  COALESCE(r.reopen_expl, 0)  AS reopen_expl,
  c.cycle_days,
  COALESCE(t.puntos, 0)       AS puntos,
  COALESCE(t.prioridad, 'Media') AS prioridad
FROM tarea t
LEFT JOIN dias_sin d ON d.tarea_id = t.id
LEFT JOIN back_moves b ON b.tarea_id = t.id
LEFT JOIN reopen_from_done r ON r.tarea_id = t.id
LEFT JOIN cycle c ON c.tarea_id = t.id
WHERE t.proyecto_id = :proyectoId;
"""

def _prioridad_num(s):
    if s == "Alta": return 3
    if s == "Baja": return 1
    return 2

def load_features(project_id:int):
    eng = get_engine()
    df = pd.read_sql(text(SQL_FEATURES), eng, params={"proyectoId": project_id})
    if df.empty:
        return df, pd.DataFrame(), []
    df["cycle_days"] = df["cycle_days"].fillna(df["cycle_days"].median())
    df["prioridad_num"] = df["prioridad"].map(_prioridad_num).fillna(2).astype(float)
    for c in ["dias_sin_mov","back_moves","reopen_expl","cycle_days","puntos","prioridad_num"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    feats = ["dias_sin_mov","back_moves","reopen_expl","cycle_days","puntos","prioridad_num"]
    X = df[feats].astype(float)
    return df, X, feats

def _build_recommendation(row, p90_dias, p90_cycle):
    motivos = []
    if row["dias_sin_mov"] >= p90_dias: motivos.append("estancada")
    if row["reopen_expl"] >= 2: motivos.append("reopen_excesivo")
    if row["back_moves"] >= 3: motivos.append("ping_pong")
    if row["cycle_days"] >= p90_cycle: motivos.append("ciclo_largo")
    if "estancada" in motivos:
        sug = "subir_prioridad o partir_tarea"
    elif "reopen_excesivo" in motivos:
        sug = "agregar_revisor o partir_tarea"
    elif "ciclo_largo" in motivos:
        sug = "partir_tarea o reforzar_equipo"
    elif "ping_pong" in motivos:
        sug = "clarificar_alcance o pair_programming"
    else:
        sug = "revisar"
    return motivos, sug

def run_anomaly_scan(project_id: int, contamination: float = 0.1):
    df, X, feats = load_features(project_id)
    if df.empty or len(df) < 8:
        return []

    p90_dias = float(np.percentile(X["dias_sin_mov"], 90))
    p90_cycle = float(np.percentile(X["cycle_days"], 90))

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    iforest = IsolationForest(n_estimators=200, contamination=contamination, random_state=42)
    lbl = iforest.fit_predict(Xs)          # -1 = anómala
    score = iforest.decision_function(Xs)  # más bajo = más raro

    df["is_anomaly"] = (lbl == -1).astype(int)
    df["score"] = score

    items = []
    for _, r in df.iterrows():
        if int(r["is_anomaly"]) != 1:
            continue
        motivos, sugerencia = _build_recommendation(r, p90_dias, p90_cycle)
        items.append({
            "tarea_id": int(r["tarea_id"]),
            "score": float(r["score"]),
            "motivos": motivos,
            "sugerencia": sugerencia,
            "resumen": {
                "dias_sin_mov": int(r["dias_sin_mov"]),
                "back_moves": int(r["back_moves"]),
                "reopen_expl": int(r["reopen_expl"]),
                "cycle_days": float(r["cycle_days"]),
                "prioridad": str(r["prioridad"]),
                "puntos": int(r["puntos"])
            }
        })

    items = sorted(items, key=lambda x: x["score"])  # más anómala primero
    return items[:50]
