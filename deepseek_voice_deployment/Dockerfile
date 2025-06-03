# DeepSeek Voice Agent - Docker Deployment
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies for audio processing
RUN apt-get update && apt-get install -y \
    portaudio19-dev \
    python3-pyaudio \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY deepseek_voice_app.py .
COPY .env.example .

# Create a non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Run the application
CMD ["streamlit", "run", "deepseek_voice_app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]

