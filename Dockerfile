FROM python:3.9-slim

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

# Ensure environment variables are available
ENV COINGECKO_API_KEY=$COINGECKO_API_KEY

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
