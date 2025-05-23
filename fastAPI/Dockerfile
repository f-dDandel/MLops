FROM python:3.10-slim
WORKDIR /house_app
COPY . .
RUN pip install -r requirements.txt
EXPOSE 8005
CMD ["python", "./main.py"]
