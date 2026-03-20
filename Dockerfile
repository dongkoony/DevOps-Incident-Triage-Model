FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV MODEL_PATH=/app/models/devops-incident-triage

RUN pip install --no-cache-dir uv

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
RUN uv pip install --system ".[api]"

COPY models ./models

EXPOSE 8000
CMD ["uvicorn", "devops_incident_triage.api:app", "--host", "0.0.0.0", "--port", "8000"]
