# 1/6: Retrieve Python 3.6.10 - Debian 10 Docker Image
FROM python:3.6.10-buster

# 2/6: Create Working Directory
WORKDIR /mamak-flask

# 3/7 [GIT]: Retrieve Source Code from GitHub
#RUN git clone --single-branch --branch develop-flask https://github.com/aqmalafiq/webMamakRecog.git

# 3/7 [LOCAL]: Copy Project Source Code Locally
ADD . /mamak-flask/

# 4/6: Install Python Project Requirements
RUN pip install --no-cache-dir --requirement requirements.txt

# 5/6: Expose Server Port to Host
EXPOSE 8000

# 6/6: Setup Container Default Command
ENTRYPOINT gunicorn -w 1 -k eventlet -b 0.0.0.0:8000 app:app