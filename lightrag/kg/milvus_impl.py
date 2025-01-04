import asyncio
import os
from tqdm.asyncio import tqdm as tqdm_async
from dataclasses import dataclass
import numpy as np
from lightrag.utils import logger
from ..base import BaseVectorStorage

from pymilvus import MilvusClient


@dataclass
class MilvusVectorDBStorage(BaseVectorStorage):
    @staticmethod
    def create_collection_if_not_exist(
        client: MilvusClient, collection_name: str, **kwargs
    ):
        if client.has_collection(collection_name):
            return
        client.create_collection(
            collection_name, max_length=64, id_type="string", **kwargs
        )

    @staticmethod
    def create_database_if_not_exist(client: MilvusClient, db_name: str):
        try:
            client.list_databases()
        except:
            # Si la liste des bases de données échoue, on se connecte d'abord à la base par défaut
            client = MilvusClient(
                uri=os.environ.get("MILVUS_URI"),
                db_name=""  # Base de données par défaut
            )
        
        databases = client.list_databases()
        if db_name not in databases:
            client.create_database(db_name)

    def __post_init__(self):
        # D'abord, créer la base de données si nécessaire
        milvus_uri = os.environ.get(
            "MILVUS_URI",
            os.path.join(self.global_config["working_dir"], "milvus_lite.db"),
        )
        logger.debug(f"Configuration Milvus - URI: {milvus_uri}")
        
        temp_client = MilvusClient(
            uri=milvus_uri,
            db_name=""  # Base de données par défaut
        )
        db_name = os.environ.get("MILVUS_DB_NAME", "")
        logger.debug(f"Configuration Milvus - DB Name: {db_name}")
        
        self.create_database_if_not_exist(temp_client, db_name)

        # Ensuite, se connecter à la base de données créée
        self._client = MilvusClient(
            uri=milvus_uri,
            user=os.environ.get("MILVUS_USER", ""),
            password=os.environ.get("MILVUS_PASSWORD", ""),
            token=os.environ.get("MILVUS_TOKEN", ""),
            db_name=db_name,
        )
        logger.debug(f"Configuration Milvus - Collection: {self.namespace}, Dimension: {self.embedding_func.embedding_dim}")
        
        self._max_batch_size = self.global_config["embedding_batch_num"]
        MilvusVectorDBStorage.create_collection_if_not_exist(
            self._client,
            self.namespace,
            dimension=self.embedding_func.embedding_dim,
        )
        
        # Vérifier les collections après création
        collections = self._client.list_collections()
        logger.debug(f"Collections disponibles après initialisation : {collections}")

    async def upsert(self, data: dict[str, dict]):
        logger.debug(f"Inserting {len(data)} vectors to {self.namespace}")
        if not len(data):
            logger.warning("You insert an empty data to vector DB")
            return []
        list_data = [
            {
                "id": k,
                **{k1: v1 for k1, v1 in v.items() if k1 in self.meta_fields},
            }
            for k, v in data.items()
        ]
        contents = [v["content"] for v in data.values()]
        batches = [
            contents[i : i + self._max_batch_size]
            for i in range(0, len(contents), self._max_batch_size)
        ]
        embedding_tasks = [self.embedding_func(batch) for batch in batches]
        embeddings_list = []
        for f in tqdm_async(
            asyncio.as_completed(embedding_tasks),
            total=len(embedding_tasks),
            desc="Generating embeddings",
            unit="batch",
        ):
            embeddings = await f
            embeddings_list.append(embeddings)
        embeddings = np.concatenate(embeddings_list)
        for i, d in enumerate(list_data):
            d["vector"] = embeddings[i]
        results = self._client.upsert(collection_name=self.namespace, data=list_data)
        return results

    async def query(self, query, top_k=5, vdb_filter=None):
        embedding = await self.embedding_func([query])
        

        # Construire l'expression de filtrage si des IDs de vdb_filter fournis
        if vdb_filter:
            filter_expr = 'id in [' + ', '.join(f'"{id}"' for id in vdb_filter) + ']'
            logger.info(f"🔍 Expression de filtrage : {filter_expr}")
        else:
            filter_expr = ""
            logger.error("🌐 Aucun filtre d'ID spécifié, exportation de toute la collection")

        #logger.info(f"Début de la recherche - vdb_filter: {vdb_filter}")
        
        if vdb_filter==None:
            logger.info("Aucun filtre de nœud spécifié, recherche sans filtrage")
            results = self._client.search(
                collection_name=self.namespace,
                data=embedding,
                limit=top_k,
                output_fields=list(self.meta_fields),
                search_params={"metric_type": "COSINE", "params": {"radius": 0.2}},
            )
        else:
            #logger.info(f"Filtrage de nœuds actif - Expression de filtre: {filter_expr}")
            #logger.info(f"collecion name: {self.namespace}")
            # Récupérer les données avec l'expression de filtrage
            
            results = self._client.search(
                collection_name=self.namespace,
                data=embedding,
                filter=filter_expr,  # Changement de 'expr' à 'filter'
                anns_field="vector",
                output_fields=list(self.meta_fields),
                #output_fields=["id", "entity_name", "entity_type"],
                limit=top_k  # Ajustez selon la taille de votre collection
            )
        
        logger.info(f"Résultats de recherche - Nombre de résultats: {len(results[0])}")
        return [
            {**dp["entity"], "id": dp["id"], "distance": dp["distance"]}
            for dp in results[0]
        ]