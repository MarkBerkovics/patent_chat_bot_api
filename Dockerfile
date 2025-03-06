FROM python:3.10.16-bookworm

COPY api /api
COPY requirements.txt /requirements.txt
COPY vector_store /vector_store
COPY sqlite3_fix.py /sqlite3_fix.py

RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN python sqlite3_fix.py

CMD uvicorn api.chat_bot_api:app --host 0.0.0.0 --port $PORT
