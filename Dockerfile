# Minimal image to run the SMS spam detector webapp
FROM python:3.13-slim

WORKDIR /app

# Copy project files
COPY . /app

# No external dependencies; if added later, uncomment:
# COPY requirements.txt /app
# RUN pip install --no-cache-dir -r requirements.txt

ENV PORT=8000
EXPOSE 8000

CMD ["python", "webapp.py"]
