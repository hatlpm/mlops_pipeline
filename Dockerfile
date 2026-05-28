# Usamos una imagen base ligera de Python 3.12
FROM python:3.12-slim

# Evita que Python escriba archivos .pyc y fuerza el output en consola
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Establecemos el directorio principal de trabajo dentro del contenedor
WORKDIR /app

# 1. Copiamos los requerimientos y los instalamos
# Esto se hace primero para aprovechar la memoria caché de Docker
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 2. Copiamos los artefactos pesados (cambian poco)
COPY model.pkl ft_pipeline.pkl ./

# 3. Copiamos el código fuente (suele cambiar más seguido)
COPY src/ ./src/

# Exponemos el puerto que usará uvicorn
EXPOSE 8000

# Nos movemos a src/ para que uvicorn encuentre fácilmente model_deploy.py
WORKDIR /app/src

# Comando para arrancar el servidor en todas las interfaces (0.0.0.0)
CMD ["uvicorn", "model_deploy:app", "--host", "0.0.0.0", "--port", "8000"]