# Minimal container for the FastAPI backend (phase 8 proof).
# The manual corpus + vector store are NOT baked in — mount data/ at runtime:
#   docker run -p 8000:8000 -e OPENAI_API_KEY=... -v "$PWD/data:/app/data" factory-floor-api
FROM python:3.13-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY factory_floor/ ./factory_floor/
COPY api/ ./api/
COPY machines.csv maintenance_history.csv manual_sources.csv fault_codes.csv operators.csv tenants.csv ./

EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
