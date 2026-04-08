FROM python:3.12-slim

WORKDIR /app

COPY processor/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY processor/ /app/processor/
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

ENV PYTHONPATH="/app"

ENTRYPOINT ["/app/entrypoint.sh"]

CMD ["python", "-m", "processor.main"]
