FROM python:3.12
#FROM --platform=linux/amd64 rotator:latest

COPY requirements.txt /app/requirements.txt
COPY src /app

WORKDIR /app


RUN pip install -r requirements.txt
EXPOSE 8010

CMD ["python", "app.py"]