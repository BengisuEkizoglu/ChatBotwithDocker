FROM python:3.8

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ./topical_chat.csv /app/topical_chat.csv

COPY . /app

EXPOSE 5000

CMD ["python", "app.py"]
