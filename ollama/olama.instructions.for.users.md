# Ollama using Docker.Compose

/binsh

```bash
version: '3.8'
services:
  ollama:
    image: ollama/ollama:latest
    container_name: ollama
    ports:
      - "11434:11434"
    volumes:
      - ./ollama:/root/.ollama
    restart: unless-stopped
    command: serve
```
Create a docker-compose.yml file with above content

```bash
docker-compose up -d
```
Access olama on the exposed port.

## To load Model using olama docker 

- get inside Ollama docker 

```bash
docker exec -it ollama /bin/bash 
```
- pull and load a model 

```bash
ollama pull mistral:instruct 
ollama pull llama3.2:1b

ollama pull llama2:7b
```

- lists all available model

```bash
ollama list
```
- remove a model

```bash
ollama rm llama3
```
You can use any other method to load load a model.