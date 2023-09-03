# Use the official lightweight Python image.
# https://hub.docker.com/_/python
FROM python:3.11-slim-buster
# Copy local code to the container image.
ENV APP_HOME /app
WORKDIR $APP_HOME
COPY . ./

# Install production dependencies.
RUN pip install --no-cache-dir -r requirements.txt

# Expose the container
EXPOSE 8080

# Run the web service on container startup.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]