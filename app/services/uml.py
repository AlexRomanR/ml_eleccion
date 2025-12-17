import io
import os
import json
import base64
import requests
import numpy as np
import cv2
import keras_ocr
from dotenv import load_dotenv

load_dotenv()

# Inicializar pipeline globalmente para eficiencia
# Se usa un try/except para evitar errores si no hay GPU o fallan las descargas iniciales en este entorno,
# aunque idealmente debería estar listo.
try:
    _pipeline = keras_ocr.pipeline.Pipeline()
except Exception as e:
    print(f"Warning: Could not initialize keras_ocr pipeline: {e}")
    _pipeline = None

# --- Helper Function for Java Code Generation ---
def generate_java_boilerplate(classes_data):
    """
    Genera el código Java (string) a partir de la lista estructurada de clases.
    Soporta la estructura [{'name': '...', 'attributes': [...], 'methods': [...]}]
    """
    code_lines = []
    
    for cls in classes_data:
        class_name = cls.get('name', 'UnknownClass').replace(" ", "")
        attributes = cls.get('attributes', [])
        methods = cls.get('methods', [])
        
        code_lines.append(f"public class {class_name} {{")
        
        # Attributes
        for attr in attributes:
            # Limpieza básica para asegurar formato válido
            attr_clean = attr.replace(" ", "")
            # Heurística simple de tipos: si no tiene tipo, asumir String
            if ":" in attr_clean:
                name, type_ = attr_clean.split(":")[:2]
                code_lines.append(f"    private {type_} {name};")
            else:
                code_lines.append(f"    private String {attr_clean};")
        
        code_lines.append("")

        # Methods
        for method in methods:
            method_clean = method.replace(" ", "")
            if not method_clean.endswith(")"):
                method_clean += "()"
            code_lines.append(f"    public void {method_clean} {{")
            code_lines.append("        // TODO: Implement method")
            code_lines.append("    }")
            code_lines.append("")
        
        code_lines.append("}")
        code_lines.append("") # Separador entre clases

    return "\n".join(code_lines)


class UMLGeneratorDeepLearning:
    def __init__(self):
        self.pipeline = _pipeline

    def process(self, image_bytes):
        """
        Método principal para procesar la imagen con enfoque local (OpenCV + KerasOCR).
        """
        if self.pipeline is None:
            return {"error": "Keras OCR pipeline not initialized"}

        # 1. Preprocesamiento OpenCV
        # Convertir bytes a imagen numpy
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return {"error": "Could not decode image"}

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Threshold para detectar cajas negras sobre fondo blanco o viceversa
        # Asumimos fondo claro y lineas oscuras (dibujo pizarra)
        # Usamos adaptiveThreshold para pizarras
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY_INV, 11, 2)

        # Detectar contornos
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        class_boxes = []
        
        min_area = 1000 # Filtrar ruido
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > min_area:
                # Aproximar polígono
                peri = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
                
                # Buscamos rectángulos (4 lados)
                if len(approx) == 4:
                    x, y, w, h = cv2.boundingRect(approx)
                    # Recortar ROI
                    roi = image[y:y+h, x:x+w]
                    class_boxes.append(roi)

        # Si no detectamos cajas rectangulares, quizás es solo texto suelto o dibujo libre.
        # Fallback: Usar toda la imagen como una sola "caja" si no hay rectángulos claros
        if not class_boxes:
            class_boxes = [image]

        # 2. OCR Extraction con keras_ocr
        detected_classes = []
        
        try:
            # Procesar cada caja detectada
            # keras_ocr espera lista de imágenes, podemos pasar todas las ROIs a la vez para batching
            # Convertir de BGR (OpenCV) a RGB (KerasOCR expects RGB normally via tools.read, but numpy array is fine if RGB)
            # Pero cv2.imdecode da BGR. Keras-ocr usa matplotlib o similar que espera RGB? 
            # documentation uses keras_ocr.tools.read which handles reading.
            # Pipeline.recognize takes numpy arrays. It expects RGB usually? Let's convert to be safe.
            rgb_boxes = [cv2.cvtColor(box, cv2.COLOR_BGR2RGB) for box in class_boxes]
            
            prediction_groups = self.pipeline.recognize(rgb_boxes)
            
            for predictions in prediction_groups:
                if not predictions:
                    continue
                
                # Heurística:
                # Ordenar por posición Y (altura)
                # Palabra más alta (menor Y) -> Nombre Clase
                # Resto -> Atributos/Métodos
                
                # predictions es lista de tuplas (text, box)
                # box es array de 4 puntos. Usamos np.mean de Y para ordenar.
                sorted_preds = sorted(predictions, key=lambda x: np.mean(x[1][:, 1]))
                
                class_name = sorted_preds[0][0]
                
                attributes = []
                methods = []
                
                for text, _ in sorted_preds[1:]:
                    # Detectar si es método por parentesis
                    if "(" in text or ")" in text:
                        methods.append(text)
                    else:
                        attributes.append(text)
                
                detected_classes.append({
                    "name": class_name,
                    "attributes": attributes,
                    "methods": methods
                })

        except Exception as e:
            print(f"Error in OCR: {e}")
            return {"error": str(e)}

        # 3. Generar Código
        java_code = generate_java_boilerplate(detected_classes)

        return {
            "method": "DeepLearning_Local",
            "classes": detected_classes,
            "generated_code": java_code
        }


class UMLGeneratorGemini:
    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY")

    def process(self, image_bytes):
        """
        Método principal para procesar la imagen con Gemini API.
        """
        if not self.api_key:
            return {"error": "Missing GEMINI_API_KEY"}

        base_url = 'https://generativelanguage.googleapis.com'
        model = 'gemini-2.0-flash' # Usando la versión recomendada/más rápida
        url = f"{base_url}/v1beta/models/{model}:generateContent?key={self.api_key}"

        # Prompt del Sistema
        system_prompt = """
        Actúa como un Ingeniero de Software experto en UML y Java.
        Tu tarea es analizar la imagen proporcionada, identificar los diagramas de clases UML (dibujados a mano o digitales)
        y extraer la estructura de las clases.
        
        Debes identificar:
        1. Nombre de la Clase
        2. Atributos (campos)
        3. Métodos (funciones)
        
        Retorna la respuesta ESTRICTAMENTE en formato JSON plano (sin markdown ```json ```).
        El JSON debe ser una LISTA de objetos, donde cada objeto representa una clase con keys: "name", "attributes", "methods".
        Los atributos y métodos deben ser listas de strings.
        
        Ejemplo de formato esperado:
        [
            {
                "name": "Usuario",
                "attributes": ["nombre: String", "edad: int"],
                "methods": ["login()", "logout()"]
            }
        ]
        Si la imagen no es clara o no se detecta nada, devuelve una lista vacía [].
        """

        try:
            # Codificar imagen
            b64_image = base64.b64encode(image_bytes).decode('utf-8')
            
            payload = {
                "contents": [{
                    "parts": [
                        {"text": system_prompt},
                        {
                            "inlineData": {
                                "mimeType": "image/jpeg", # Asumimos jpeg/png genérico
                                "data": b64_image
                            }
                        }
                    ]
                }],
                "generationConfig": {
                    "temperature": 0.1, # Bajo para mayor determinismo y apego al formato
                    "responseMimeType": "application/json" # Forzar JSON output mode si está disponible en 2.0-flash
                }
            }
            
            response = requests.post(url, json=payload, headers={"Content-Type": "application/json"})
            
            if response.status_code != 200:
                print(f"Gemini API Error: {response.text}")
                return {"error": f"API Error {response.status_code}", "details": response.text}
            
            data = response.json()
            candidates = data.get('candidates', [])
            
            if not candidates:
                 return {"method": "Gemini_API", "classes": [], "generated_code": "", "error": "No candidates returned"}

            text_response = candidates[0]['content']['parts'][0]['text']
            
            # Limpieza adicional por seguridad (aunque generationConfig debería ayudar)
            clean_text = text_response.strip()
            if clean_text.startswith("```json"):
                clean_text = clean_text[7:]
            if clean_text.startswith("```"):
                clean_text = clean_text[3:]
            if clean_text.endswith("```"):
                clean_text = clean_text[:-3]
            
            classes_data = json.loads(clean_text)
            
            # Generar código con el mismo helper para consistencia
            java_code = generate_java_boilerplate(classes_data)
            
            return {
                "method": "Gemini_API",
                "classes": classes_data,
                "generated_code": java_code
            }

        except Exception as e:
            return {"error": str(e), "method": "Gemini_API"}

def main():
    # Ejemplo de uso
    image_path = "diagrama_test.jpg"  # Reemplazar con una ruta real para probar
    
    print(f"--- Iniciando UML Extractor ---")
    if not os.path.exists(image_path):
        print(f"No se encontró la imagen de prueba: {image_path}")
        print("Crea una imagen llamada 'diagrama_test.jpg' en este directorio para probar.")
        # Creamos una imagen dummy negra con texto blanco para que no falle el script si el usuario lo corre
        # O simplemente retornamos.
        return

    with open(image_path, "rb") as f:
        img_bytes = f.read()

    # 1. Probar Deep Learning
    print("\n[1] Ejecutando Extractor Local (Deep Learning)...")
    dl_extractor = UMLGeneratorDeepLearning()
    dl_result = dl_extractor.process(img_bytes)
    
    if "error" in dl_result:
        print(f"Error Local: {dl_result['error']}")
    else:
        print(f"Detectado: {len(dl_result['classes'])} clases.")
        print("Clases:", [c['name'] for c in dl_result['classes']])
        print("Código Generado (Snippet):")
        print(dl_result['generated_code'][:200] + "...\n")

    # 2. Probar Gemini
    print("\n[2] Ejecutando Extractor Gemini (LLM)...")
    gemini_extractor = UMLGeneratorGemini()
    gemini_result = gemini_extractor.process(img_bytes)

    if "error" in gemini_result:
        print(f"Error Gemini: {gemini_result.get('details', gemini_result['error'])}")
    else:
        print(f"Detectado: {len(gemini_result['classes'])} clases.")
        print("Clases:", [c['name'] for c in gemini_result['classes']])
        print("Código Generado (Snippet):")
        print(gemini_result['generated_code'][:200] + "...\n")

if __name__ == "__main__":
    main()
