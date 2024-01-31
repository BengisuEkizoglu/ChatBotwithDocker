# Chatbot with Docker Container

Containerize the ChatBot Chroma application

## Installation

Get the Chroma Docker image from Docker Hub to run a Chroma server in a Docker container 

```bash
docker pull chromadb/chroma
docker run -p 8000:8000 chromadb/chroma
```

After the installation of all files, run the command below on the path of docker-compose.yml file.

```bash
docker-compose up --build 
