
# Stage 1: build Node dependencies
FROM node:18-slim AS node-build
WORKDIR /app
COPY serve_requirements package*.json ./
RUN npm ci --omit=dev

# Stage 2: runtime
FROM python:3.11-slim

# Install minimal OS deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
RUN pip install --no-cache-dir -r serve_requirements.txt

# Copy Node runtime deps from builder
COPY --from=node-build /app/node_modules ./node_modules
COPY --from=node-build /app/package*.json ./

# Copy only necessary source files (not the whole repo)
COPY server.js ./
COPY public/ ./public/
COPY model_and_encoder.joblib /app/model_and_encoder.joblib

EXPOSE 3000
CMD ["node", "server.js"]