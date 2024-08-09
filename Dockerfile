FROM python:3

WORKDIR /app

COPY app.py .

RUN pip install Flask

CMD ["python", "app.py"]
