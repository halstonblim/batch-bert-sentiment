FROM python:3.12-slim

RUN apt-get update && apt-get install -y git

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .

ENV REPLICATE_PREDICTOR=predict:Predictor

CMD ["python3"]
