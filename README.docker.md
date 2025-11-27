# Guide Docker pour le Workshop DevFest 2025

## Utilisation avec Docker

### Option 1 : Utiliser Ollama local (recommandé)

Si vous avez déjà Ollama installé sur votre machine :

```bash
# Construire l'image
docker compose build app

# Lancer un shell interactif
docker compose run --rm app /bin/bash

# Dans le container, vous pouvez exécuter les scripts :
python3 01_local_llm/hello_world.py
python3 02_rag_lcel/ingest.py
python3 02_rag_lcel/query.py --interactive
```

### Option 2 : Utiliser Ollama dans Docker

Si vous n'avez pas Ollama installé localement :

```bash
# Lancer Ollama et l'app
docker compose up -d ollama

# Attendre quelques secondes puis télécharger les modèles
docker exec ollama-server ollama pull llama3.1:latest
docker exec ollama-server ollama pull nomic-embed-text

# Optionnel : modèle de réflexion
docker exec ollama-server ollama pull qwen3:8b

# Modifier le docker-compose.yml pour que l'app utilise le service ollama
# Remplacer "network_mode: host" par "depends_on: ollama"
# et ajouter dans environment : OLLAMA_HOST=http://ollama:11434

# Lancer l'application
docker compose run --rm app /bin/bash
```

### Exécuter des scripts directement

```bash
# Hello World
docker compose run --rm app python3 01_local_llm/hello_world.py

# RAG Ingestion
docker compose run --rm app python3 02_rag_lcel/ingest.py

# RAG Query
docker compose run --rm app python3 02_rag_lcel/query.py --question "Who is the CEO?"

# Agent ReAct
docker compose run --rm app python3 03_langgraph_react/agent.py --interactive

# Supervisor
docker compose run --rm app python3 04_supervisor/supervisor.py

# Network
docker compose run --rm app python3 05_network/network.py
```

### Variables d'environnement

Créez un fichier `.env` à la racine du projet si vous utilisez des modèles cloud :

```bash
GOOGLE_API_KEY=votre_cle_api
```

### Nettoyage

```bash
# Arrêter tous les conteneurs
docker compose down

# Supprimer les volumes (attention : efface les données)
docker compose down -v
```

## Notes

- Les données persistantes sont stockées dans le dossier `./data`
- Le fichier `.env` est monté en lecture seule
- Le mode `network_mode: host` permet à l'app d'accéder à Ollama sur `localhost:11434`
- Pour utiliser un GPU avec Ollama, assurez-vous d'avoir installé nvidia-docker
