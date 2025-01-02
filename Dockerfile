# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy the current directory contents into the container
COPY . .

RUN pip install --upgrade pip

# Install any needed packages specified in requirements.txt
RUN pip install -r requirements.txt

RUN chmod u+rwx models

RUN chmod -R 755 models


# Run app.py when the container launches
CMD ["gunicorn", "-b", "0.0.0.0:7860", "app:app"]
