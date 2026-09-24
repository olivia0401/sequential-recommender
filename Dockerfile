FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY api ./api

# The trained checkpoint is not in the repo (*.pt is gitignored). Mount it at
# runtime, e.g.:
#   docker run -p 8000:8000 -v "$PWD/model_retailrocket.pt:/models/model.pt:ro" seqrec
ENV MODEL_PATH=/models/model.pt

EXPOSE 8000
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
