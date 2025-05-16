FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Copy the application code into the container
COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 5000

CMD ["python", "app.py"]