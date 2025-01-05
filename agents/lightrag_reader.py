# Standard library imports
import asyncio
import json
import logging
import os
import re
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

# Third-party library imports
import aio_pika
import numpy as np
import requests
from dotenv import load_dotenv
from requests.exceptions import HTTPError, Timeout

# Phi-related imports
from phi.agent import Agent
from phi.knowledge.agent import AgentKnowledge
from phi.model.openai import OpenAIChat
from phi.storage.agent.postgres import PgAgentStorage
from phi.vectordb.pgvector import PgVector, SearchType

# Local project imports
from agents.settings import agent_settings
from db.session import db_url
from lightrag import LightRAG, QueryParam
from lightrag.llm import gpt_4o_mini_complete, gpt_4o_complete
from lightrag.utils import EmbeddingFunc
from lightrag.prompt import PROMPTS
from lightrag.kg.mongo_impl import MongoKVStorage
from lightrag.kg.milvus_impl import MilvusVectorDBStorage



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
    
def lightrag_query(query: str = "une recherche dans la base graph"):
    try:
        rag = init_lightrag()
        
        # Préparation des paramètres de requête
        mode="hybrid"
        query_param = QueryParam(mode=mode)
        
        #vdb_filter= [ "Zulli"]
        vdb_filter= ["lea"]

        # Exécution asynchrone de la requête
        response = asyncio.run(rag.aquery(query, param=query_param, vdb_filter=vdb_filter))
        
        print(f"\nQuestion: {query}")
        print(f"\nRéponse: {response}")

    except Exception as e:
        print(f"An error occurred: {e}")




def get_lightrag_reader(
    model_id: Optional[str] = None,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    debug_mode: bool = False,
) -> Agent:
    return Agent(
        name="LIGHTRAG Query",
        agent_id="lightrag_query",
        session_id=session_id,
        user_id=user_id,
        # The model to use for the agent
        model=OpenAIChat(
            id=model_id or agent_settings.gpt_4,
            max_tokens=agent_settings.default_max_completion_tokens,
            temperature=agent_settings.default_temperature,
        ),
        # model=Ollama(id="qwen2.5m:latest"),
        # Tools available to the agent
        tools=[lightrag_query],
        # A description of the agent that guides its overall behavior
        role="Tu es un agent d'une Team qui dispose des capacités à aider l'utilisateur à trouver une information dans la base graph.",  
        # A list of instructions to follow, each as a separate item in the list
        instructions=[
            "Etape 1 : Analyser la demande et utiliser uniquement les informations en provenance de la fonction lightrag_query pour répondre à l'utilisateur.\n",
            ],
        # Format responses as markdown
        markdown=True,
        # Show tool calls in the response
        show_tool_calls=True,
        # Add the current date and time to the instructions
        add_datetime_to_instructions=True,
        # Store agent sessions in the database
        #storage=lightrag_reader_storage,
        # Enable read the chat history from the database
        #read_chat_history=True,
        # Store knowledge in a vector database
        #knowledge=lightrag_reader_knowledge,
        # Enable searching the knowledge base
        #search_knowledge=True,
        # Enable monitoring on phidata.app
        monitoring=True,
        # Show debug logs
        debug_mode=True,
    )