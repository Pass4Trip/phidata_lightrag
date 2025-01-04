import os
import logging
from neo4j import AsyncGraphDatabase

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Neo4jSubgraphExtractor:
    def __init__(self, uri=None, username=None, password=None):
        """
        Initialise la connexion à Neo4j
        """
        URI = uri or os.environ["NEO4J_URI"]
        USERNAME = username or os.environ["NEO4J_USERNAME"]
        PASSWORD = password or os.environ["NEO4J_PASSWORD"]
        
        logger.info(f"Connexion à Neo4j avec l'URI : {URI}")
        
        self._driver = AsyncGraphDatabase.driver(URI, auth=(USERNAME, PASSWORD))

    async def extract_subgraph(self, custom_ids):
        """
        Extrait un sous-graphe pour une liste de custom_ids
        
        Returns:
            dict: Un dictionnaire structuré similaire à knowledge_graph_inst
        """
        async with self._driver.session() as session:
            query = """
            MATCH (n)
            WHERE n.custom_id IN $custom_ids
            MATCH (n)-[r]-(connected)
            RETURN 
                n AS source_node, 
                r AS relationship, 
                connected AS target_node,
                labels(n) AS source_labels,
                labels(connected) AS target_labels
            """
            
            result = await session.run(query, {"custom_ids": custom_ids})
            
            # Structure pour stocker les entités et relations
            entities = {}
            relations = []
            
            async for record in result:
                # Traitement du nœud source
                source_id = record["source_node"].element_id
                if source_id not in entities:
                    entities[source_id] = {
                        "id": source_id,
                        "labels": record["source_labels"],
                        "properties": dict(record["source_node"])
                    }
                
                # Traitement du nœud cible
                target_id = record["target_node"].element_id
                if target_id not in entities:
                    entities[target_id] = {
                        "id": target_id,
                        "labels": record["target_labels"],
                        "properties": dict(record["target_node"])
                    }
                
                # Ajout de la relation
                relations.append({
                    "source_id": source_id,
                    "target_id": target_id,
                    "type": record["relationship"].type,
                    "properties": dict(record["relationship"])
                })
            
            logger.info(f"Sous-graphe extrait pour {len(custom_ids)} nœuds")
            return {
                "entities": list(entities.values()),
                "relations": relations
            }

    def close(self):
        """
        Ferme la connexion au driver Neo4j
        """
        if self._driver:
            self._driver.close()

from lightrag.lightrag import LightRAG
from dotenv import load_dotenv
import os
import asyncio
import logging

async def main():
    # Configuration du logging
    logging.basicConfig(level=logging.INFO)
    
    # Charger les variables d'environnement
    load_dotenv()

    # Initialiser LightRAG
    lightrag = LightRAG(graph_storage="Neo4JStorage")

    # Liste des custom_ids à extraire (à ajuster selon vos données)
    custom_ids = ["5390255707819795563"]

    try:
        # Extraction synchrone du sous-graphe
        sync_subgraph = lightrag.extract_subgraph(custom_ids)
        print("Sous-graphe synchrone :")
        print(sync_subgraph)

        # Extraction asynchrone du sous-graphe
        async_subgraph = await lightrag.aextract_subgraph(custom_ids)
        print("\nSous-graphe asynchrone :")
        print(async_subgraph)

    except Exception as e:
        logging.error(f"Erreur lors de l'extraction du sous-graphe : {e}")

if __name__ == "__main__":
    asyncio.run(main())