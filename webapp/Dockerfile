# Utiliser une image Python officielle comme base
FROM python:3.8-slim

# Spécifier le répertoire de travail dans le conteneur
WORKDIR /app

# Copier les dépendances du projet dans le conteneur
COPY requirements.txt requirements.txt

# Installer les dépendances du projet
RUN pip install --no-cache-dir -r requirements.txt

# Copier les fichiers de l'application dans le conteneur
COPY . .

# Exposer le port sur lequel l'application s'exécutera
EXPOSE 7860

# Commande pour démarrer l'application lorsque le conteneur est lancé
CMD ["python", "web_app.py"]