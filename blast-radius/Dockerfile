
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY run.py /app/run.py
COPY select_tests.py /app/select_tests.py
COPY scope_lookup.py /app/scope_lookup.py
COPY skip_rules.py /app/skip_rules.py
COPY code_graph/ /app/code_graph/

RUN chmod -R a+rX /app

USER 65534:65534

ENTRYPOINT ["python3", "/app/run.py"]
