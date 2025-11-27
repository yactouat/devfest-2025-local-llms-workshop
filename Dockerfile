FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies including SQLite
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    libsqlite3-dev \
    sqlite3 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
# Remove problematic packages that are not available on all platforms
RUN grep -v -E "(pysqlite3-binary|sqlite-vss)" requirements.txt > /tmp/requirements.txt && \
    echo "chromadb" >> /tmp/requirements.txt && \
    pip install --no-cache-dir -r /tmp/requirements.txt

# Copy application code
COPY . .

# Create volume mount point for persistent data
VOLUME ["/app/data"]

# Default command (can be overridden)
CMD ["python3", "01_local_llm/hello_world.py"]
