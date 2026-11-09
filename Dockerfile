FROM python:3.13-slim
WORKDIR /app
COPY deployment/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt fastapi uvicorn
COPY deployment/ ./deployment/
COPY app.py ./
EXPOSE 8000
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}"]
