FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

WORKDIR /app-gradio

RUN pip install --no-cache-dir \
    gradio==3.12.0 \
    tensorflow==2.12.0 \
    pillow==10.0.0 \
    wandb==0.15.5 \
    numpy==1.25.0

COPY grad.py /app-gradio/

EXPOSE 7860

CMD ["python", "grad.py"]
