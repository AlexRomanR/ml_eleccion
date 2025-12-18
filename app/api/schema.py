# app/api/schema.py
import base64
import typing
from functools import lru_cache

import strawberry
from strawberry.scalars import JSON


# ---------------------------
# Lazy singletons (cacheados)
# ---------------------------

@lru_cache(maxsize=1)
def _get_uml_local():
    # ✅ Importa recién cuando se usa (evita cargar TensorFlow en el arranque)
    from app.services.uml import UMLGeneratorDeepLearning
    return UMLGeneratorDeepLearning()

@lru_cache(maxsize=1)
def _get_uml_gemini():
    from app.services.uml import UMLGeneratorGemini
    return UMLGeneratorGemini()

@lru_cache(maxsize=1)
def _get_kanban_extractor():
    from app.services.kanban import KanbanExtractor
    return KanbanExtractor()

def _run_anomaly_scan(project_id: int, contamination: float):
    # ✅ También lo hacemos lazy (pandas/sklearn)
    from app.services.anomalies import run_anomaly_scan
    return run_anomaly_scan(project_id, contamination)


# ---------------------------
# Types
# ---------------------------

@strawberry.type
class UMLClass:
    name: str
    attributes: typing.List[str]
    methods: typing.List[str]

@strawberry.type
class UMLResponse:
    method: str
    classes: typing.List[UMLClass]
    generated_code: str
    error: typing.Optional[str] = None

@strawberry.type
class AnomalySummary:
    dias_sin_mov: int
    back_moves: int
    reopen_expl: int
    cycle_days: float
    prioridad: str
    puntos: int

@strawberry.type
class AnomalyItem:
    tarea_id: int
    titulo: typing.Optional[str]  
    score: float
    motivos: typing.List[str]
    sugerencia: str
    resumen: AnomalySummary

@strawberry.type
class ScanResponse:
    project_id: int
    anomalies: typing.List[AnomalyItem]


# ---------------------------
# Query / Mutation
# ---------------------------

@strawberry.type
class Query:
    @strawberry.field
    def health(self) -> bool:
        return True

    @strawberry.field
    def scan_anomalies(self, project_id: int, contamination: float = 0.1) -> ScanResponse:
        results = _run_anomaly_scan(project_id, contamination)

        anomalies_objs: typing.List[AnomalyItem] = []
        for item in results:
            summary_data = item["resumen"]
            anomalies_objs.append(
                AnomalyItem(
                    tarea_id=item["tarea_id"],
                    titulo=item.get("titulo"),   # ✅ nuevo
                    score=item["score"],
                    motivos=item["motivos"],
                    sugerencia=item["sugerencia"],
                    resumen=AnomalySummary(
                        dias_sin_mov=summary_data["dias_sin_mov"],
                        back_moves=summary_data["back_moves"],
                        reopen_expl=summary_data["reopen_expl"],
                        cycle_days=summary_data["cycle_days"],
                        prioridad=summary_data["prioridad"],
                        puntos=summary_data["puntos"],
                    ),
                )
            )

        return ScanResponse(project_id=project_id, anomalies=anomalies_objs)


@strawberry.type
class Mutation:
    @strawberry.mutation
    def generate_uml(self, image_base64: str, use_gemini: bool = False) -> UMLResponse:
        try:
            if "," in image_base64:
                image_base64 = image_base64.split(",")[1]
            image_bytes = base64.b64decode(image_base64)

            # ✅ se instancia recién aquí
            if use_gemini:
                result = _get_uml_gemini().process(image_bytes)
            else:
                result = _get_uml_local().process(image_bytes)

            if result.get("error"):
                return UMLResponse(
                    method=result.get("method", "Unknown"),
                    classes=[],
                    generated_code="",
                    error=result["error"],
                )

            classes_objs: typing.List[UMLClass] = []
            for cls_data in result.get("classes", []):
                classes_objs.append(
                    UMLClass(
                        name=cls_data.get("name", "Unknown"),
                        attributes=cls_data.get("attributes", []),
                        methods=cls_data.get("methods", []),
                    )
                )

            return UMLResponse(
                method=result.get("method", "Unknown"),
                classes=classes_objs,
                generated_code=result.get("generated_code", ""),
                error=None,
            )
        except Exception as e:
            return UMLResponse(
                method="Error",
                classes=[],
                generated_code="",
                error=f"Internal Server Error: {str(e)}",
            )

    @strawberry.mutation
    def extract_kanban(self, image_base64: str) -> JSON:
        try:
            if "," in image_base64:
                image_base64 = image_base64.split(",")[1]
            image_bytes = base64.b64decode(image_base64)

            # ✅ se instancia recién aquí
            return _get_kanban_extractor().extract_from_bytes(image_bytes)
        except Exception as e:
            return {"error": str(e)}


schema = strawberry.Schema(query=Query, mutation=Mutation)
