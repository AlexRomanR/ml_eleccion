import strawberry
import base64
import typing
import json
from strawberry.scalars import JSON
from app.services.uml import UMLGeneratorDeepLearning, UMLGeneratorGemini
from app.services.kanban import KanbanExtractor
from app.services.anomalies import run_anomaly_scan

# Instanciar servicios
uml_local = UMLGeneratorDeepLearning()
uml_gemini = UMLGeneratorGemini()
kanban_extractor = KanbanExtractor()

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
    score: float
    motivos: typing.List[str]
    sugerencia: str
    resumen: AnomalySummary

@strawberry.type
class ScanResponse:
    project_id: int
    anomalies: typing.List[AnomalyItem]

@strawberry.type
class Query:
    @strawberry.field
    def health(self) -> bool:
        return True

    @strawberry.field
    def scan_anomalies(self, project_id: int, contamination: float = 0.1) -> ScanResponse:
        results = run_anomaly_scan(project_id, contamination)
        
        # Mapear diccionarios a objetos
        anomalies_objs = []
        for item in results:
            summary_data = item['resumen']
            anomalies_objs.append(AnomalyItem(
                tarea_id=item['tarea_id'],
                score=item['score'],
                motivos=item['motivos'],
                sugerencia=item['sugerencia'],
                resumen=AnomalySummary(
                    dias_sin_mov=summary_data['dias_sin_mov'],
                    back_moves=summary_data['back_moves'],
                    reopen_expl=summary_data['reopen_expl'],
                    cycle_days=summary_data['cycle_days'],
                    prioridad=summary_data['prioridad'],
                    puntos=summary_data['puntos']
                )
            ))
            
        return ScanResponse(project_id=project_id, anomalies=anomalies_objs)

@strawberry.type
class Mutation:
    @strawberry.mutation
    def generate_uml(self, image_base64: str, use_gemini: bool = False) -> UMLResponse:
        try:
            if "," in image_base64:
                image_base64 = image_base64.split(",")[1]
            image_bytes = base64.b64decode(image_base64)
            
            if use_gemini:
                result = uml_gemini.process(image_bytes)
            else:
                result = uml_local.process(image_bytes)
            
            if "error" in result and result.get("error"):
                return UMLResponse(
                    method=result.get("method", "Unknown"),
                    classes=[],
                    generated_code="",
                    error=result["error"]
                )

            classes_objs = []
            for cls_data in result.get("classes", []):
                classes_objs.append(UMLClass(
                    name=cls_data.get("name", "Unknown"),
                    attributes=cls_data.get("attributes", []),
                    methods=cls_data.get("methods", [])
                ))

            return UMLResponse(
                method=result.get("method", "Unknown"),
                classes=classes_objs,
                generated_code=result.get("generated_code", ""),
                error=None
            )
        except Exception as e:
            return UMLResponse(
                method="Error",
                classes=[],
                generated_code="",
                error=f"Internal Server Error: {str(e)}"
            )

    @strawberry.mutation
    def extract_kanban(self, image_base64: str) -> JSON:
        """
        Retorna la estructura del tablero Kanban como un objeto JSON arbitrario
        debido a la complejidad y profundidad variable de la respuesta.
        """
        try:
            if "," in image_base64:
                image_base64 = image_base64.split(",")[1]
            image_bytes = base64.b64decode(image_base64)
            
            return kanban_extractor.extract_from_bytes(image_bytes)
        except Exception as e:
            return {"error": str(e)}

schema = strawberry.Schema(query=Query, mutation=Mutation)
