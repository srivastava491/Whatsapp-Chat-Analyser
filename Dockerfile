FROM python:3.10-slim-buster

WORKDIR /app

RUN apt update -y && apt install -y git

COPY requirements.txt /app/

RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

EXPOSE 8501
CMD ["streamlit", "run", "app.py"]
