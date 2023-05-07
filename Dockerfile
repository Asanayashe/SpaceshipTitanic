FROM python:3.8-slim-buster

COPY *.py /app/
COPY spaceship_transformed.csv /app/
COPY requirements.txt /app/

WORKDIR /app/

RUN pip install -r requirements.txt

RUN python train.py

EXPOSE 80

# Run uvicorn server
CMD ["uvicorn", "server:app", "--reload", "--host", "0.0.0.0", "--port", "80"]
