FROM python:3.6
COPY requirements.txt /requirements.txt
RUN pip install -r requirements.txt
EXPOSE 5000
COPY . /.
CMD ["python", "hello.py"]
