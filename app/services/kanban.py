import io
import os
import json
import base64
import requests
import numpy as np
import keras_ocr
from PIL import Image
from sklearn.cluster import DBSCAN
from dotenv import load_dotenv

load_dotenv()

# Initialize pipeline once
_pipeline = keras_ocr.pipeline.Pipeline()

class KanbanExtractor:
    def __init__(self):
        self.pipeline = _pipeline
        self.gemini_key = os.getenv("GEMINI_API_KEY")

    def extract_from_bytes(self, image_bytes):
        # 1. Try Local Extraction
        print("Starting Local OCR extraction...")
        local_result = self._extract_local(image_bytes)
        
        # 2. Validation / Quality Check
        # Heuristic: If we found less than 2 columns or 0 tasks, it's probably wrong.
        col_count = len(local_result['columns'])
        total_tasks = sum(len(c['tasks']) for c in local_result['columns'])
        
        is_poor_quality = col_count < 2 or total_tasks == 0
        
        if is_poor_quality:
            print(f"Local OCR incomplete (Cols: {col_count}, Tasks: {total_tasks}). Attempting Fallback...")
            if self.gemini_key:
                try:
                    return self._call_gemini_fallback(image_bytes)
                except Exception as e:
                    print(f"Gemini Fallback failed: {e}")
                    # If fallback fails, return local result anyway (better than crash)
                    return local_result
            else:
                print("No GEMINI_API_KEY found. Returning local result.")
        
        return local_result

    def _extract_local(self, image_bytes):
        image = keras_ocr.tools.read(io.BytesIO(image_bytes))
        prediction_groups = self.pipeline.recognize([image])
        predictions = prediction_groups[0]

        if not predictions:
            return self._empty_structure()

        elements = []
        for text, box in predictions:
            center = np.mean(box, axis=0)
            elements.append({
                'text': text,
                'center_x': center[0],
                'center_y': center[1],
                'height': np.max(box[:, 1]) - np.min(box[:, 1]),
                'box': box
            })

        return self._structure_data(elements, image.shape[1], image.shape[0])

    def _structure_data(self, elements, width, height):
        header_thresh = height * 0.15
        header_elems = [e for e in elements if e['center_y'] < header_thresh]
        body_elems = [e for e in elements if e['center_y'] >= header_thresh]

        project_name = "Proyecto Detectado"
        if header_elems:
            header_elems.sort(key=lambda x: x['height'], reverse=True)
            project_name = header_elems[0]['text']
            if len(header_elems) > 1:
                 project_name += " " + header_elems[1]['text']

        if not body_elems:
             return self._empty_structure(project_name)

        x_coords = np.array([e['center_x'] for e in body_elems]).reshape(-1, 1)
        
        # DBSCAN
        eps = width * 0.15 
        clustering = DBSCAN(eps=eps, min_samples=1).fit(x_coords)
        
        columns_map = {} 
        for idx, label in enumerate(clustering.labels_):
            if label not in columns_map: columns_map[label] = []
            columns_map[label].append(body_elems[idx])

        sorted_cols = []
        for label, elems in columns_map.items():
            avg_x = np.mean([e['center_x'] for e in elems])
            sorted_cols.append({'elems': elems, 'x': avg_x})
        
        sorted_cols.sort(key=lambda c: c['x'])

        final_columns = []
        for i, col_data in enumerate(sorted_cols):
            elems = col_data['elems']
            elems.sort(key=lambda e: e['center_y'])
            
            col_name = elems[0]['text']
            card_elems = elems[1:]

            tasks = []
            if card_elems:
                current_card_texts = []
                last_y = card_elems[0]['center_y']
                last_h = card_elems[0]['height']
                
                for e in card_elems:
                    gap = e['center_y'] - last_y
                    threshold = max(last_h, e['height']) * 2.0 
                    
                    if gap > threshold and current_card_texts:
                        tasks.append(self._parse_card(current_card_texts))
                        current_card_texts = []

                    current_card_texts.append(e)
                    last_y = e['center_y']
                    last_h = e['height']
                
                if current_card_texts:
                    tasks.append(self._parse_card(current_card_texts))

            final_columns.append({
                "nombre": col_name,
                "orden": i + 1,
                "tasks": tasks
            })

        return {
            "project": {
                "nombre": project_name,
                "descripcion": None,
                "estado": "Activo",
                "fechaInicio": None,
                "fechaFin": None
            },
            "columns": final_columns
        }

    def _parse_card(self, elements):
        full_text = " ".join([e['text'] for e in elements])
        prioridad = "MEDIA"
        if "alta" in full_text.lower(): prioridad = "ALTA"
        elif "baja" in full_text.lower(): prioridad = "BAJA"
        
        return {
            "titulo": elements[0]['text'], 
            "descripcion": " ".join([e['text'] for e in elements[1:]]) if len(elements)>1 else None,
            "prioridad": prioridad,
            "puntos": 0,
            "fechaLimite": None
        }

    def _empty_structure(self, name="Sin Nombre"):
        return {
            "project": {"nombre": name, "estado": "Activo"},
            "columns": []
        }

    # --- GEMINI FALLBACK LOGIC ---
    def _call_gemini_fallback(self, image_bytes):
        base_url = 'https://generativelanguage.googleapis.com'
        model = 'gemini-2.0-flash' # Or 1.5-flash, trying a robust one
        url = f"{base_url}/v1beta/models/{model}:generateContent?key={self.gemini_key}"
        
        prompt = '''
        Eres un extractor OCR + estructurador de tableros Kanban.
        A partir de la imagen, detecta: nombre del proyecto, columnas y tareas.
        Responde SOLO en JSON válido con este esquema:
        {
          "project": { "nombre": "string", "descripcion": "string", "estado": "Activo", "fechaInicio": "string", "fechaFin": "string" },
          "columns": [
            { "nombre": "string", "orden": 1, "tasks": [ { "titulo": "string", "descripcion": "string", "prioridad": "BAJA|MEDIA|ALTA", "puntos": 0, "fechaLimite": "YYYY-MM-DD" } ] }
          ]
        }
        '''
        
        payload = {
            "contents": [{
                "parts": [
                    {"text": prompt},
                    {
                        "inlineData": {
                            "mimeType": "image/jpeg",
                            "data": base64.b64encode(image_bytes).decode('utf-8')
                        }
                    }
                ]
            }]
        }
        
        resp = requests.post(url, json=payload, headers={"Content-Type": "application/json"})
        if resp.status_code != 200:
            raise Exception(f"Gemini API Error {resp.status_code}: {resp.text}")
            
        return self._parse_gemini_response(resp.json())

    def _parse_gemini_response(self, data):
        try:
            candidates = data.get('candidates', [])
            if not candidates: return self._empty_structure("Error Gemini")
            
            text_part = candidates[0]['content']['parts'][0]['text']
            # Strip fences
            cleaned = text_part.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[-1].rsplit("\n", 1)[0]
            if cleaned.startswith("json"):
                cleaned = cleaned[4:].strip()
                
            return json.loads(cleaned)
        except Exception as e:
            print(f"Error parsing Gemini JSON: {e}")
            return self._empty_structure("Error Parsing")
