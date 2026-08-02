FROM python:3.12.5-slim@sha256:9c756fe68086f8fb720977a05ef0d1931d0c2fc9ad6a22caadd7f4ce0d8b7417 AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app
RUN addgroup --system app && adduser --system --ingroup app app
COPY pyproject.toml README.md ./
COPY src ./src
RUN python -m pip install --upgrade pip && python -m pip install ".[vision]"
USER app
EXPOSE 8080
CMD ["uvicorn", "phone_dino.app:app", "--host", "0.0.0.0", "--port", "8080", "--no-access-log"]
