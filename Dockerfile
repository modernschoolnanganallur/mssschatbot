# Use lightweight Python image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements first (better caching)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Ensure script is executable
RUN chmod +x start.sh

# Expose port for Render or local
EXPOSE 10000

# Start command
CMD ["./start.sh"]
