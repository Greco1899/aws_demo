# Base image to use
FROM python:3.9-slim

# Expose ST port
EXPOSE 8080

# Copy and install requirements
COPY requirements.txt app/requirements.txt
RUN pip install -r app/requirements.txt

# Copy all files to app directory
COPY . /app

# Change working directory
WORKDIR /app

# Run application
ENTRYPOINT ["streamlit", "run", "gcp_app.py", "--server.port=8080", "--server.address=0.0.0.0"]

