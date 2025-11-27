FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src/ ./src/
RUN pip install --no-cache-dir -e .
EXPOSE 8082
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8082/health || exit 1
CMD ["uvicorn", "sam3_agent.main:app", "--host", "0.0.0.0", "--port", "8082"]
