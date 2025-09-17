FROM python:3.11-slim

# Install Node.js and build tools
RUN apt-get update && apt-get install -y curl build-essential ca-certificates \
  && curl -fsSL https://deb.nodesource.com/setup_18.x | bash - \
  && apt-get install -y nodejs && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency files first (better caching)
COPY requirements.txt package*.json ./

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install Node.js dependencies
RUN npm install --production

# Copy project source code
COPY . .

# Copy the trained model explicitly (artifact from CI)
COPY model_and_encoder.joblib /app/model_and_encoder.joblib

EXPOSE 3000
CMD ["node", "server.js"]