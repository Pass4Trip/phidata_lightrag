import os
import sys
import logging
import re
from typing import Dict, Any
import json
import asyncio
from pathlib import Path
from dotenv import load_dotenv
import uuid
from datetime import datetime
import traceback
import time

# Load environment variables from .env file
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add local LightRAG source to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from lightrag.lightrag import LightRAG, QueryParam
from lightrag.llm import gpt_4o_mini_complete, gpt_4o_complete
from lightrag.utils import EmbeddingFunc
from lightrag.prompt import PROMPTS
from lightrag.kg.mongo_impl import MongoKVStorage
from lightrag.kg.milvus_impl import MilvusVectorDBStorage



# Configuration Milvus - utiliser les valeurs de .env ou les valeurs par défaut
if not os.environ.get("MILVUS_URI"):
    os.environ["MILVUS_URI"] = "tcp://localhost:19530"

    
def init_lightrag():
    """
    Initialise LightRAG avec MongoDB, Neo4j et Milvus
    Utilise les variables d'environnement pour les connexions
    """
    working_dir = "./data"
    
    # Création du répertoire de travail s'il n'existe pas
    if not os.path.exists(working_dir):
        os.makedirs(working_dir)
        logger.debug(f"Répertoire de travail créé: {working_dir}")
    
    try:
        # Initialisation de LightRAG
        rag = LightRAG(
            working_dir=working_dir,
            llm_model_func=gpt_4o_mini_complete,
            kv_storage="MongoKVStorage",      # MongoDB pour le stockage clé-valeur
            vector_storage="MilvusVectorDBStorage",  # Milvus pour les vecteurs
            graph_storage="Neo4JStorage",     # Neo4j pour le graphe
            log_level="INFO",
            enable_llm_cache=False  # Ajout du paramètre ici
        )
        logger.debug("LightRAG initialisé avec succès")
        return rag
    except Exception as e:
        logger.error(f"Erreur lors de l'initialisation de LightRAG: {str(e)}")
        raise

def query_lightrag(question: str, mode: str = "hybrid"):
    """
    Interroge LightRAG avec une question
    
    Args:
        question (str): La question à poser
        mode (str): Mode de recherche ('naive', 'local', 'global', 'hybrid')
    
    Returns:
        str: La réponse générée
    """
    try:
        rag = init_lightrag()
        logger.info(f"Question posée: {question}")
        response = rag.query(question, param=QueryParam(mode=mode))
        logger.info("Réponse générée avec succès")
        return response
    except Exception as e:
        logger.error(f"Erreur lors de la requête: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        # Exemple d'utilisation
        question = "dis moi ce que tu sais sur lea"
        rag = init_lightrag()
        
        # Exemple d'utilisation avec filtrage
        # node_list = [{'custom_id': 'ZUlli'}]
        

        # Préparation des paramètres de requête
        mode="hybrid"
        query_param = QueryParam(mode=mode)
        
        #vdb_filter= [ "Zulli"]
        vdb_filter= ["lea"]

        # Exécution asynchrone de la requête
        response = asyncio.run(rag.aquery(question, param=query_param, vdb_filter=vdb_filter))
        
        print(f"\nQuestion: {question}")
        print(f"\nRéponse: {response}")
    except Exception as e:
        print(f"An error occurred: {e}")