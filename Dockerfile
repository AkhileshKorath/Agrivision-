# Use official lightweight Node.js LTS image
FROM node:20-slim

# Install Python 3, pip, and essential build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Python requirements
COPY requirements.txt ./
RUN pip3 install --no-cache-dir -r requirements.txt

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
