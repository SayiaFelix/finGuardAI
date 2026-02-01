# lightweight Python image
FROM python:3.9-slim-buster
# python:3.9-slim
# Set the working directory inside the container
WORKDIR /app
 
# Copy application code to the container
COPY . /app
 
# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN apt-get update && apt-get install -y libgomp1 libglib2.0-0
 
# Expose the application
EXPOSE 5001
 
# The command to run the application
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5001", "fin_guard_ai:app"]