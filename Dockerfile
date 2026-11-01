# Use official lightweight Node.js LTS image
FROM node:20-slim

# Install Python 3, pip, venv, and libgomp1 (required for onnxruntime runtime)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Create and activate Python virtual environment (fixes Debian 12 PEP 668 externally-managed-environment)
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
ENV PIP_BREAK_SYSTEM_PACKAGES=1

# Install Python requirements inside virtualenv
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Install Node.js dependencies
COPY package*.json ./
RUN npm install --omit=dev

# Copy application code
COPY . .

# Ensure uploads directory exists
RUN mkdir -p uploads public

# Set environment
ENV PORT=3000
EXPOSE 3000

# Start server
CMD ["node", "server.js"]
