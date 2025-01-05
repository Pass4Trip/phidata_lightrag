## Agent Api

This repo contains the code for running an agent-api and supports 2 environments:

1. **dev**: A development environment running locally on docker
2. **prd**: A production environment running on AWS ECS

## Configuration et Installation

### Prérequis
- Python 3.12
- Docker
- Environnement virtuel (recommandé)

### Étapes d'installation

1. **Configurer l'environnement**
```bash
# Créer un environnement virtuel
python3.12 -m venv .venv
source .venv/bin/activate

# Installer les dépendances
pip install -r requirements.txt

# Installer phidata en mode éditable
pip install -e ///app/phidata
```

2. **Configuration OpenAI**
- Ajouter votre clé API OpenAI dans `.env`
```
OPENAI_API_KEY=sk-votre_clé_api
```

3. **Démarrer les conteneurs Docker**
```bash
# Créer et démarrer les conteneurs
phi ws up -y

# Arrêter les conteneurs
phi ws down -y
```

4. **Résolution des dépendances**
Si des erreurs de dépendances surviennent :
```bash
pip install aioboto3 hnswlib pika prefect pymilvus pymongo pyvis tenacity xxhash
```

### Dépannage
- Vérifiez que tous les packages sont installés
- Assurez-vous que la clé API OpenAI est correcte
- Utilisez `--verbose` pour plus d'informations lors de l'installation

## Run Agent Api locally

1. Install [docker desktop](https://www.docker.com/products/docker-desktop)

2. Set OpenAI Key

Set the `OPENAI_API_KEY` environment variable using

```sh
export OPENAI_API_KEY=sk-***
```

**OR** set in the `.env` file

3. Start the workspace using:

```sh
phi ws up
```

Open [localhost:8000/docs](http://localhost:8000/docs) to view the FastAPI docs.

4. Stop the workspace using:

```sh
phi ws down
```

## Next Steps:

- [Run the Agent Api on AWS](https://docs.phidata.com/templates/agent-api/run-aws)
- Read how to [manage the development application](https://docs.phidata.com/how-to/development-app)
- Read how to [manage the production application](https://docs.phidata.com/how-to/production-app)
- Read how to [add python libraries](https://docs.phidata.com/how-to/python-libraries)
- Read how to [format & validate your code](https://docs.phidata.com/how-to/format-and-validate)
- Read how to [manage secrets](https://docs.phidata.com/how-to/secrets)
- Add [CI/CD](https://docs.phidata.com/how-to/ci-cd)
- Add [database tables](https://docs.phidata.com/how-to/database-tables)
- Read the [Agent Api guide](https://docs.phidata.com/templates/agent-api)
