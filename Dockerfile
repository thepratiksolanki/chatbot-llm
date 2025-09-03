FROM python:3.10-slim

WORKDIR /app
COPY . .

RUN pip install --no-cache-dir -r requirements.txt

ENV PORT=8080
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 app:app