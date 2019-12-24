FROM python:3.6
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
ENV $FLASK_APP app.py
CMD ["flask","run"]

