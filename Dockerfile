# Use a lightweight Python image
FROM python:3.10.19

USER root

# Set working directory
WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# ============================================
# Install System Dependencies
# ============================================
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    poppler-utils \
    tesseract-ocr \
    libmagic1 \
    libglib2.0-0 \
    libgl1 \
    wget \
    gnupg \
    unzip \
    curl \
    ca-certificates \
    libnss3 \
    libfontconfig1 \
    libxss1 \
    libayatana-appindicator3-1 \
    libasound2 \
    libatk-bridge2.0-0 \
    libgtk-3-0 \
    libx11-xcb1 \
    libxcomposite1 \
    libxcursor1 \
    libxdamage1 \
    libxi6 \
    libxtst6 \
    xdg-utils \
    fonts-liberation \
    s3cmd \
  && rm -rf /var/lib/apt/lists/*

# ============================================
# Install Google Chrome
# ============================================
RUN apt-get update && apt-get install -y --no-install-recommends \
        wget gnupg ca-certificates \
    && wget -q -O /tmp/google-linux-signing-key.pub \
        https://dl.google.com/linux/linux_signing_key.pub \
    && gpg --dearmor -o /usr/share/keyrings/google-linux-signing-keyring.gpg \
        /tmp/google-linux-signing-key.pub \
    && echo "deb [arch=amd64 signed-by=/usr/share/keyrings/google-linux-signing-keyring.gpg] \
        http://dl.google.com/linux/chrome/deb/ stable main" \
        > /etc/apt/sources.list.d/google-chrome.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends google-chrome-stable \
    && rm -rf /var/lib/apt/lists/* /tmp/google-linux-signing-key.pub

# ============================================
# Install Python Dependencies
# ============================================
COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt

# ============================================
# Copy Application Code
# ============================================
COPY . .

# ============================================
# Create app user and directories
# ============================================
RUN addgroup --system app && adduser --system --ingroup app app && \
    mkdir -p /app/nltk_data && \
    mkdir -p /app/hf_cache && \
    mkdir -p /app/chromedriver && \
    mkdir -p /app/.wdm && \
    mkdir -p /app/logs && \
    mkdir -p /app/web_scrap && \
    mkdir -p /app/tmp

# ============================================
# Install ChromeDriver in /app/chromedriver
# ============================================
RUN CHROME_VERSION=$(google-chrome --version | awk '{print $3}') && \
    echo "Chrome version: ${CHROME_VERSION}" && \
    wget -q -O /tmp/chromedriver-linux64.zip \
    "https://storage.googleapis.com/chrome-for-testing-public/${CHROME_VERSION}/linux64/chromedriver-linux64.zip" && \
    unzip /tmp/chromedriver-linux64.zip -d /tmp/ && \
    mv /tmp/chromedriver-linux64/chromedriver /app/chromedriver/chromedriver && \
    chmod +x /app/chromedriver/chromedriver && \
    rm -rf /tmp/chromedriver-linux64.zip /tmp/chromedriver-linux64

# ============================================
# Set ownership to app user
# ============================================
RUN chown -R app:app /app

# ============================================
# Environment Variables
# ============================================
ENV NLTK_DATA=/app/nltk_data
ENV HF_HOME=/app/hf_cache
ENV TRANSFORMERS_CACHE=/app/hf_cache
ENV HOME=/app
ENV TMPDIR=/app/tmp
ENV WDM_LOCAL=1
ENV PATH="/app/chromedriver:${PATH}"

# ============================================
# Switch to app user
# ============================================
USER app

EXPOSE 8095

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8095"]
#CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8095", "--workers", "1", "--limit-concurrency", "15", "--backlog", "100"]
#CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "app:app", "--bind", "0.0.0.0:8095", "--workers", "1", "--timeout", "300", "--graceful-timeout", "30", "--keep-alive", "5", "--access-logfile", "-", "--error-logfile", "-", "--log-level", "info"]
