FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

WORKDIR /app-api

RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    tensorflow \
    pillow \
    wandb \
    numpy

COPY fastapi_app.py /app-api/

EXPOSE 8000

# Command to run FastAPI
CMD ["uvicorn", "fastapi_app:app", "--host", "0.0.0.0", "--port", "8000"]
