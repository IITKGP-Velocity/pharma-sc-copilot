FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir -e ".[app]"
EXPOSE 8501
# Pass the university values at run time, never inside the image:
#   docker run --env-file .env -e COPILOT_CHAT_API_KEY -v "$PWD/secrets:/app/secrets:ro" -p 8501:8501 pharma-copilot
CMD ["python", "-m", "streamlit", "run", "app/app.py", "--server.address=0.0.0.0", "--server.headless=true"]
