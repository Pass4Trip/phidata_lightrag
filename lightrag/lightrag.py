import asyncio
import os
from tqdm.asyncio import tqdm as tqdm_async
from dataclasses import asdict, dataclass, field
from datetime import datetime
from functools import partial
from typing import Type, cast, Optional, List, Dict, Any

from lightrag.llm import (
    gpt_4o_mini_complete,
    openai_embedding,
)
from lightrag.operate import (
    chunking_by_token_size,
    extract_entities,
    # local_query,global_query,hybrid_query,
    kg_query,
    naive_query,
)

from lightrag.utils import (
    EmbeddingFunc,
    compute_mdhash_id,
    limit_async_func_call,
    convert_response_to_json,
    logger,
    set_logger,
)
from lightrag.base import (
    BaseGraphStorage,
    BaseKVStorage,
    BaseVectorStorage,
    StorageNameSpace,
    QueryParam,
)

from lightrag.storage import (
    JsonKVStorage,
    NanoVectorDBStorage,
    NetworkXStorage,
)

from lightrag.kg.neo4j_impl import Neo4JStorage

#from lightrag.kg.oracle_impl import OracleKVStorage, OracleGraphStorage, OracleVectorDBStorage

from lightrag.kg.milvus_impl import MilvusVectorDBStorage

from lightrag.kg.mongo_impl import MongoKVStorage

# future KG integrations

# from lightrag.kg.ArangoDB_impl import (
#     GraphStorage as ArangoDBStorage
# )


def always_get_an_event_loop() -> asyncio.AbstractEventLoop:
    """
    Ensure that there is always an event loop available.

    This function tries to get the current event loop. If the current event loop is closed or does not exist,
    it creates a new event loop and sets it as the current event loop.

    Returns:
        asyncio.AbstractEventLoop: The current or newly created event loop.
    """
    try:
        # Try to get the current event loop
        current_loop = asyncio.get_event_loop()
        if current_loop._closed:
            raise RuntimeError("Event loop is closed.")
        return current_loop

    except RuntimeError:
        # If no event loop exists or it is closed, create a new one
        logger.debug("Creating a new event loop in main thread.")
        new_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(new_loop)
        return new_loop


@dataclass
class LightRAG:
    working_dir: str = field(
        default_factory=lambda: f"./lightrag_cache_{datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}"
    )
    # Default not to use embedding cache
    embedding_cache_config: dict = field(
        default_factory=lambda: {
            "enabled": False,
            "similarity_threshold": 0.95,
            "use_llm_check": False,
        }
    )
    kv_storage: str = field(default="JsonKVStorage")
    vector_storage: str = field(default="NanoVectorDBStorage")
    graph_storage: str = field(default="NetworkXStorage")

    current_log_level = logger.level
    log_level: str = field(default=current_log_level)

    # text chunking
    chunk_token_size: int = 1200
    chunk_overlap_token_size: int = 100
    tiktoken_model_name: str = "gpt-4o-mini"

    # entity extraction
    entity_extract_max_gleaning: int = 1
    entity_summary_to_max_tokens: int = 500

    # node embedding
    node_embedding_algorithm: str = "node2vec"
    node2vec_params: dict = field(
        default_factory=lambda: {
            "dimensions": 1536,
            "num_walks": 10,
            "walk_length": 40,
            "window_size": 2,
            "iterations": 3,
            "random_seed": 3,
        }
    )

    # embedding_func: EmbeddingFunc = field(default_factory=lambda:hf_embedding)
    embedding_func: EmbeddingFunc = field(default_factory=lambda: openai_embedding)
    embedding_batch_num: int = 32
    embedding_func_max_async: int = 16

    # LLM
    llm_model_func: callable = gpt_4o_mini_complete  # hf_model_complete#
    llm_model_name: str = "meta-llama/Llama-3.2-1B-Instruct"  #'meta-llama/Llama-3.2-1B'#'google/gemma-2-2b-it'
    llm_model_max_token_size: int = 32768
    llm_model_max_async: int = 16
    llm_model_kwargs: dict = field(default_factory=dict)

    # storage
    vector_db_storage_cls_kwargs: dict = field(default_factory=dict)

    enable_llm_cache: bool = True

    # extension
    addon_params: dict = field(default_factory=dict)
    convert_response_to_json_func: callable = convert_response_to_json


    # Nouveau paramètre avec valeur par défaut
    prompt_domain: str = field(default="activity")


    def __post_init__(self):
        log_file = os.path.join("lightrag.log")
        set_logger(log_file)
        logger.setLevel(self.log_level)

        logger.debug(f"Logger initialized for working directory: {self.working_dir}")

        _print_config = ",\n  ".join([f"{k} = {v}" for k, v in asdict(self).items()])
        logger.debug(f"LightRAG init with param:\n  {_print_config}\n")

        # @TODO: should move all storage setup here to leverage initial start params attached to self.

        self.key_string_value_json_storage_cls: Type[BaseKVStorage] = (
            self._get_storage_class()[self.kv_storage]
        )
        self.vector_db_storage_cls: Type[BaseVectorStorage] = self._get_storage_class()[
            self.vector_storage
        ]
        self.graph_storage_cls: Type[BaseGraphStorage] = self._get_storage_class()[
            self.graph_storage
        ]

        if not os.path.exists(self.working_dir):
            logger.debug(f"Creating working directory {self.working_dir}")
            os.makedirs(self.working_dir)

        self.llm_response_cache = (
            self.key_string_value_json_storage_cls(
                namespace="llm_response_cache",
                global_config=asdict(self),
                embedding_func=None,
            )
            if self.enable_llm_cache
            else None
        )
        self.embedding_func = limit_async_func_call(self.embedding_func_max_async)(
            self.embedding_func
        )

        ####
        # add embedding func by walter
        ####
        self.full_docs = self.key_string_value_json_storage_cls(
            namespace="full_docs",
            global_config=asdict(self),
            embedding_func=self.embedding_func,
        )
        self.text_chunks = self.key_string_value_json_storage_cls(
            namespace="text_chunks",
            global_config=asdict(self),
            embedding_func=self.embedding_func,
        )
        self.chunk_entity_relation_graph = self.graph_storage_cls(
            namespace="chunk_entity_relation",
            global_config=asdict(self),
            embedding_func=self.embedding_func,
        )
        ####
        # add embedding func by walter over
        ####

        self.entities_vdb = self.vector_db_storage_cls(
            namespace="entities",
            global_config=asdict(self),
            embedding_func=self.embedding_func,
            meta_fields={"entity_name", "entity_type"},
        )
        self.relationships_vdb = self.vector_db_storage_cls(
            namespace="relationships",
            global_config=asdict(self),
            embedding_func=self.embedding_func,
            meta_fields={"src_id", "tgt_id"},
        )
        self.chunks_vdb = self.vector_db_storage_cls(
            namespace="chunks",
            global_config=asdict(self),
            embedding_func=self.embedding_func,
        )

        self.llm_model_func = limit_async_func_call(self.llm_model_max_async)(
            partial(
                self.llm_model_func,
                hashing_kv=self.llm_response_cache,
                **self.llm_model_kwargs,
            )
        )

    def _get_storage_class(self) -> Type[BaseGraphStorage]:
        return {
            # kv storage
            "JsonKVStorage": JsonKVStorage,
            #"OracleKVStorage": OracleKVStorage,
            "MongoKVStorage": MongoKVStorage,
            # vector storage
            "NanoVectorDBStorage": NanoVectorDBStorage,
            #"OracleVectorDBStorage": OracleVectorDBStorage,
            "MilvusVectorDBStorage": MilvusVectorDBStorage,
            # graph storage
            "NetworkXStorage": NetworkXStorage,
            "Neo4JStorage": Neo4JStorage,
            #"OracleGraphStorage": OracleGraphStorage,
            # "ArangoDBStorage": ArangoDBStorage
        }

    def insert(self, string_or_strings):
        loop = always_get_an_event_loop()
        return loop.run_until_complete(self.ainsert(string_or_strings))

    async def ainsert(self, string_or_strings, **kwargs):
        # Récupérer le prompt_domain soit de kwargs, soit de l'instance
        prompt_domain = kwargs.get('prompt_domain', self.prompt_domain)
        # Récupérer les metadata
        metadata = kwargs.get('metadata', {})
        
        # Log du domaine de prompt utilisé
        logger.debug(f"Inserting with prompt domain: {prompt_domain}")
        
        update_storage = False
        try:
            if isinstance(string_or_strings, str):
                string_or_strings = [string_or_strings]

            new_docs = {
                compute_mdhash_id(c.strip(), prefix="doc-"): {"content": c.strip()}
                for c in string_or_strings
            }
            _add_doc_keys = await self.full_docs.filter_keys(list(new_docs.keys()))
            new_docs = {k: v for k, v in new_docs.items() if k in _add_doc_keys}
            if not len(new_docs):
                logger.warning("All docs are already in the storage")
                return
            update_storage = True
            logger.debug(f"[New Docs] inserting {len(new_docs)} docs")

            inserting_chunks = {}
            for doc_key, doc in tqdm_async(
                new_docs.items(), desc="Chunking documents", unit="doc"
            ):
                chunks = {
                    compute_mdhash_id(dp["content"], prefix="chunk-"): {
                        **dp,
                        "full_doc_id": doc_key,
                    }
                    for dp in chunking_by_token_size(
                        doc["content"],
                        overlap_token_size=self.chunk_overlap_token_size,
                        max_token_size=self.chunk_token_size,
                        tiktoken_model=self.tiktoken_model_name,
                    )
                }
                inserting_chunks.update(chunks)
            _add_chunk_keys = await self.text_chunks.filter_keys(
                list(inserting_chunks.keys())
            )
            inserting_chunks = {
                k: v for k, v in inserting_chunks.items() if k in _add_chunk_keys
            }
            if not len(inserting_chunks):
                logger.warning("All chunks are already in the storage")
                return
            logger.debug(f"[New Chunks] inserting {len(inserting_chunks)} chunks")

            await self.chunks_vdb.upsert(inserting_chunks)
            
            logger.debug("[Entity Extraction]...")
            maybe_new_kg = await extract_entities(
                inserting_chunks,
                knowledge_graph_inst=self.chunk_entity_relation_graph,
                entity_vdb=self.entities_vdb,
                relationships_vdb=self.relationships_vdb,
                global_config=asdict(self),
                prompt_domain=prompt_domain,
                metadata=metadata,
                text_chunks=self.text_chunks
            )
            if maybe_new_kg is None:
                logger.warning("No new entities and relationships found")
                return
            self.chunk_entity_relation_graph = maybe_new_kg

            await self.full_docs.upsert(new_docs)
            await self.text_chunks.upsert(inserting_chunks)
            
            # Ajout de la logique de catégorisation des activités après l'insertion
            if update_storage and self.chunk_entity_relation_graph is not None:
                from .config.activity_categories import activity_categories_manager
            
                # Catégorisation des activités
                if prompt_domain == 'activity':
                    # Log du début du processus de catégorisation
                    logger.info("🔍 Début de la catégorisation des activités")
                    
                    # Vérifier si la méthode existe et est appelable
                    if hasattr(self.chunk_entity_relation_graph, 'categorize_activities') and callable(getattr(self.chunk_entity_relation_graph, 'categorize_activities')):
                        logger.debug("✅ Méthode categorize_activities trouvée")
                        try:
                            use_model_func = self.global_config.get("llm_model_func") if hasattr(self, 'global_config') else None
                            await self.chunk_entity_relation_graph.categorize_activities(
                                activity_categories_manager, 
                                use_model_func=use_model_func
                            )
                        except Exception as e:
                            logger.error(f"❌ Erreur lors de l'appel de categorize_activities : {e}")
                    else:
                        logger.warning("❌ Méthode categorize_activities non trouvée")
                
                # Catégorisation des activités par villes
                if prompt_domain in ['activity', 'event']:
                    logger.info("🌆 Début de la catégorisation des villes")
                    
                    if hasattr(self.chunk_entity_relation_graph, 'categorize_cities') and callable(getattr(self.chunk_entity_relation_graph, 'categorize_cities')):
                        logger.info("✅ Méthode categorize_cities trouvée")
                        try:
                            # Extraire la ville des métadonnées
                            city_name = metadata.get('city')
                            
                            if city_name:
                                # Obtenir l'élément ID du nœud
                                custom_id = metadata.get('custom_id')
                                
                                if custom_id:
                                    await self.chunk_entity_relation_graph.categorize_cities(
                                        custom_id=custom_id, 
                                        city_name=city_name
                                    )
                                    logger.info(f"✅ Ville {city_name} associée")
                                else:
                                    logger.warning("❌ custom_id manquant pour la catégorisation de la ville")
                            else:
                                logger.debug("ℹ️ Pas de ville spécifiée dans les métadonnées")
                        
                        except Exception as e:
                            logger.error(f"❌ Erreur lors de l'appel de categorize_cities : {e}")
                    else:
                        logger.warning("❌ Méthode categorize_cities non trouvée")
                
                # Catégorisation des dates pour les événements
                if prompt_domain == 'event':
                    logger.info("📅 Début de la catégorisation des dates d'événements")
                    
                    if hasattr(self.chunk_entity_relation_graph, 'categorize_dates') and callable(getattr(self.chunk_entity_relation_graph, 'categorize_dates')):
                        logger.info("✅ Méthode categorize_dates trouvée")
                        try:
                            # Extraire la date de début des métadonnées
                            start_date = metadata.get('start_date')
                            
                            if start_date:
                                # Formater la date en YYYY-MM-DD
                                from datetime import datetime
                                try:
                                    parsed_date = datetime.fromisoformat(start_date.replace('+00:00', ''))
                                    formatted_date = parsed_date.strftime('%Y-%m-%d')
                                    
                                    # Obtenir l'élément ID de l'événement
                                    custom_id = metadata.get('custom_id')
                                    
                                    if custom_id:
                                        await self.chunk_entity_relation_graph.categorize_dates(
                                            custom_id=custom_id, 
                                            date_label=formatted_date
                                        )
                                        logger.info(f"✅ Date {formatted_date} associée à l'événement")
                                    else:
                                        logger.warning("❌ custom_id manquant pour la catégorisation de la date")
                                
                                except ValueError as ve:
                                    logger.error(f"❌ Erreur de formatage de date : {ve}")
                            else:
                                logger.debug("ℹ️ Pas de date spécifiée dans les métadonnées")
                        
                        except Exception as e:
                            logger.error(f"❌ Erreur lors de l'appel de categorize_dates : {e}")
                    else:
                        logger.warning("❌ Méthode categorize_dates non trouvée")
                
                # Catégorisation des mémos
                elif prompt_domain == 'memo':
                    logger.info("📝 Début de la catégorisation des mémos")
                    
                    if hasattr(self.chunk_entity_relation_graph, 'categorize_memos') and callable(getattr(self.chunk_entity_relation_graph, 'categorize_memos')):
                        logger.info("✅ Méthode categorize_memos trouvée")
                        try:
                            # Extraire l'ID du mémo des métadonnées
                            custom_id = metadata.get('custom_id')
                            
                            if custom_id:
                                # Extraire l'ID de l'utilisateur des métadonnées si disponible
                                user_id = metadata.get('user_id')
                                
                                await self.chunk_entity_relation_graph.categorize_memos(
                                    custom_id=custom_id, 
                                    user_id=user_id
                                )
                                logger.info(f"✅ Mémo {custom_id} associé")
                            else:
                                logger.warning("❌ custom_id manquant pour la catégorisation du mémo")
                        
                        except Exception as e:
                            logger.error(f"❌ Erreur lors de l'appel de categorize_memos : {e}")
                    else:
                        logger.warning("❌ Méthode categorize_memos non trouvée")
        finally:
            if update_storage:
                await self._insert_done()

    async def _insert_done(self):
        tasks = []
        for storage_inst in [
            self.full_docs,
            self.text_chunks,
            self.llm_response_cache,
            self.entities_vdb,
            self.relationships_vdb,
            self.chunks_vdb,
            self.chunk_entity_relation_graph,
        ]:
            if storage_inst is None:
                continue
            tasks.append(cast(StorageNameSpace, storage_inst).index_done_callback())
        await asyncio.gather(*tasks)

    def insert_custom_kg(self, custom_kg: dict):
        loop = always_get_an_event_loop()
        return loop.run_until_complete(self.ainsert_custom_kg(custom_kg))

    async def ainsert_custom_kg(self, custom_kg: dict):
        """
        Asynchronous method to insert custom knowledge graph data.
        
        Args:
            custom_kg: Custom knowledge graph data
        
        Returns:
            bool: Indicates if storage was updated
        """
        # Log d'entrée DÉTAILLÉ
        logger.debug("🚀 DÉBUT de ainsert_custom_kg")
        logger.debug(f"🔍 Docs reçus : {len(custom_kg.get('docs', []))}")
        logger.debug(f"🔍 Entities_data reçus : {len(custom_kg.get('entities', []))}")

        # Reste du code inchangé
        update_storage = False
        all_entities_data = []

        # Le reste de la méthode reste identique
        try:
            # Insert chunks into vector storage
            all_chunks_data = {}
            chunk_to_source_map = {}
            for chunk_data in custom_kg.get("chunks", []):
                chunk_content = chunk_data["content"]
                source_id = chunk_data["source_id"]
                chunk_id = compute_mdhash_id(chunk_content.strip(), prefix="chunk-")

                chunk_entry = {"content": chunk_content.strip(), "source_id": source_id}
                all_chunks_data[chunk_id] = chunk_entry
                chunk_to_source_map[source_id] = chunk_id
                update_storage = True

            if self.chunks_vdb is not None and all_chunks_data:
                await self.chunks_vdb.upsert(all_chunks_data)
            if self.text_chunks is not None and all_chunks_data:
                await self.text_chunks.upsert(all_chunks_data)

            # Insert entities into knowledge graph
            all_entities_data = []
            for entity_data in custom_kg.get("entities", []):
                entity_name = f'"{entity_data["entity_name"].upper()}"'
                entity_type = entity_data.get("entity_type", "UNKNOWN")
                description = entity_data.get("description", "No description provided")
                # source_id = entity_data["source_id"]
                source_chunk_id = entity_data.get("source_id", "UNKNOWN")
                source_id = chunk_to_source_map.get(source_chunk_id, "UNKNOWN")

                # Log if source_id is UNKNOWN
                if source_id == "UNKNOWN":
                    logger.warning(
                        f"Entity '{entity_name}' has an UNKNOWN source_id. Please check the source mapping."
                    )

                # Compute Milvus ID once and reuse it
                milvus_id = compute_mdhash_id(entity_data["entity_name"].upper(), prefix="ent-")
                node_data = {
                    "entity_type": entity_type,
                    "description": description,
                    "source_id": source_id,
                    "milvus_id": milvus_id
                }
                # Debug log DÉTAILLÉ
                logger.debug(f"🔍 DEBUG Before upsert_node - entity_name: {entity_name}")
                logger.debug(f"🔍 DEBUG Before upsert_node - entity_data: {entity_data}")
                logger.debug(f"🔍 DEBUG Before upsert_node - node_data: {node_data}")
                logger.debug(f"🔍 DEBUG Before upsert_node - node_data keys: {list(node_data.keys())}")
                logger.debug(f"🔍 DEBUG Before upsert_node - milvus_id: {milvus_id}")
                logger.debug(f"🔍 DEBUG Before upsert_node - milvus_id type: {type(milvus_id)}")

                # Insert node data into the knowledge graph
                await self.chunk_entity_relation_graph.upsert_node(
                    entity_name, node_data=node_data
                )
                node_data["entity_name"] = entity_name
                all_entities_data.append(node_data)
                update_storage = True

            # Insert relationships into knowledge graph
            all_relationships_data = []
            for relationship_data in custom_kg.get("relationships", []):
                src_id = f'"{relationship_data["src_id"].upper()}"'
                tgt_id = f'"{relationship_data["tgt_id"].upper()}"'
                description = relationship_data["description"]
                keywords = relationship_data["keywords"]
                weight = relationship_data.get("weight", 1.0)
                # source_id = relationship_data["source_id"]
                source_chunk_id = relationship_data.get("source_id", "UNKNOWN")
                source_id = chunk_to_source_map.get(source_chunk_id, "UNKNOWN")

                # Log if source_id is UNKNOWN
                if source_id == "UNKNOWN":
                    logger.warning(
                        f"Relationship from '{src_id}' to '{tgt_id}' has an UNKNOWN source_id. Please check the source mapping."
                    )

                # Check if nodes exist in the knowledge graph
                for need_insert_id in [src_id, tgt_id]:
                    if not (
                        await self.chunk_entity_relation_graph.has_node(need_insert_id)
                    ):
                        await self.chunk_entity_relation_graph.upsert_node(
                            need_insert_id,
                            node_data={
                                "source_id": source_id,
                                "description": "UNKNOWN",
                                "entity_type": "UNKNOWN",
                            },
                        )

                # Insert edge into the knowledge graph
                await self.chunk_entity_relation_graph.upsert_edge(
                    src_id,
                    tgt_id,
                    edge_data={
                        "weight": weight,
                        "description": description,
                        "keywords": keywords,
                        "source_id": source_id,
                    },
                )
                edge_data = {
                    "src_id": src_id,
                    "tgt_id": tgt_id,
                    "description": description,
                    "keywords": keywords,
                }
                all_relationships_data.append(edge_data)
                update_storage = True

            # Insert entities into vector storage if needed
            if self.entities_vdb is not None:
                data_for_vdb = {
                    dp["milvus_id"]: {  # Utiliser l'ID déjà calculé
                        "content": dp["entity_name"] + dp["description"],
                        "entity_name": dp["entity_name"],
                    }
                    for dp in all_entities_data
                }
                await self.entities_vdb.upsert(data_for_vdb)

            # Insert relationships into vector storage if needed
            if self.relationships_vdb is not None:
                data_for_vdb = {
                    compute_mdhash_id(dp["src_id"] + dp["tgt_id"], prefix="rel-"): {
                        "src_id": dp["src_id"],
                        "tgt_id": dp["tgt_id"],
                        "content": dp["keywords"]
                        + dp["src_id"]
                        + dp["tgt_id"]
                        + dp["description"],
                    }
                    for dp in all_relationships_data
                }
                await self.relationships_vdb.upsert(data_for_vdb)
        finally:
            if update_storage:
                await self._insert_done()

    def query(self, query: str, param: QueryParam = QueryParam()):
        loop = always_get_an_event_loop()
        return loop.run_until_complete(self.aquery(query, param))

    async def aquery(self, query: str, param: QueryParam = QueryParam(), vdb_filter: Optional[Dict[str, Any]] = None):
        if param.mode in ["local", "global", "hybrid"]:
            response = await kg_query(
                query,
                self.chunk_entity_relation_graph,
                self.entities_vdb,
                self.relationships_vdb,
                self.text_chunks,
                param,
                asdict(self),
                hashing_kv=self.llm_response_cache,
                vdb_filter=vdb_filter,
            )
        elif param.mode == "naive":
            response = await naive_query(
                query,
                self.chunks_vdb,
                self.text_chunks,
                param,
                asdict(self),
                hashing_kv=self.llm_response_cache,
            )
        else:
            raise ValueError(f"Unknown mode {param.mode}")
        await self._query_done()
        return response

    async def _query_done(self):
        tasks = []
        for storage_inst in [self.llm_response_cache]:
            if storage_inst is None:
                continue
            tasks.append(cast(StorageNameSpace, storage_inst).index_done_callback())
        await asyncio.gather(*tasks)

    def delete_by_entity(self, entity_name: str):
        loop = always_get_an_event_loop()
        return loop.run_until_complete(self.adelete_by_entity(entity_name))

    async def adelete_by_entity(self, entity_name: str):
        entity_name = f'"{entity_name.upper()}"'

        try:
            await self.entities_vdb.delete_entity(entity_name)
            await self.relationships_vdb.delete_relation(entity_name)
            await self.chunk_entity_relation_graph.delete_node(entity_name)

            logger.debug(
                f"Entity '{entity_name}' and its relationships have been deleted."
            )
            await self._delete_by_entity_done()
        except Exception as e:
            logger.error(f"Error while deleting entity '{entity_name}': {e}")

    async def _delete_by_entity_done(self):
        tasks = []
        for storage_inst in [
            self.entities_vdb,
            self.relationships_vdb,
            self.chunk_entity_relation_graph,
        ]:
            if storage_inst is None:
                continue
            tasks.append(cast(StorageNameSpace, storage_inst).index_done_callback())
        await asyncio.gather(*tasks)
