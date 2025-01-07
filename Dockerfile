# Utiliser une image Python légère
FROM python:3.9-slim

# Installer les dépendances système
RUN apt-get update && apt-get install -y git

# Définir le répertoire de travail
WORKDIR /app

# Copier les fichiers du projet
COPY . /app

# Installer les dépendances Python
RUN pip install -r requirements.txt

# Exposer le port pour l'API
EXPOSE 8000

# Lancer l'application FastAPI avec Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
