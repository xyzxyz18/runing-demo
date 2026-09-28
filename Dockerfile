FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLCONFIGDIR=/tmp/pace-matplotlib \
    PACE_DATA_DIR=/data

WORKDIR /app
RUN apt-get -o Acquire::Retries=5 update && apt-get -o Acquire::Retries=5 install -y --no-install-recommends \
    libgl1 libglib2.0-0 libx11-6 ffmpeg fonts-noto-cjk && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --timeout 60 --retries 5 -r requirements.txt

COPY app.py main.py config.py ./
COPY analysis ./analysis
COPY biomechanics ./biomechanics
COPY pose ./pose
COPY visualization ./visualization
COPY static ./static
COPY templates ./templates
RUN useradd --uid 10001 --create-home pace && mkdir -p /data /tmp/pace-matplotlib \
    && chown -R pace:pace /data /tmp/pace-matplotlib

USER pace
EXPOSE 8000
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "1", "--threads", "4", "--timeout", "300", "app:app"]
