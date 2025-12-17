# Usar imagen base ligera de Python 3.9 (compatible y estable)
FROM python:3.9-slim

# Evitar archivos .pyc y salidas en buffer
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Instalar dependencias del sistema necesarias para OpenCV y compilación básica
# libgl1-mesa-glx y libglib2.0-0 son CRÍTICAS para que cv2 funcione en Docker
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Crear directorio de trabajo
WORKDIR /app

# Copiar requirements primero para aprovechar caché de Docker
COPY requirements.txt .

# Instalar dependencias de Python
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copiar el resto del código del microservicio
COPY . .

# Exponer el puerto 5000 (donde correrá FastAPI/Uvicorn)
EXPOSE 5000

# Comando de arranque: Uvicorn sirviendo la app "app" desde "app.py"
# Se usa --host 0.0.0.0 para que sea accesible desde fuera del contenedor
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "5000"]
