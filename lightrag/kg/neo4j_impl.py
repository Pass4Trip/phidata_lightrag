import asyncio
import os
import re
from dataclasses import dataclass
from typing import Any, Union, Tuple, List, Dict, Optional
import inspect
from lightrag.utils import logger
from ..base import BaseGraphStorage
from neo4j import (
    AsyncGraphDatabase,
    exceptions as neo4jExceptions,
    AsyncDriver,
    AsyncManagedTransaction,
)


from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)


@dataclass
class Neo4JStorage(BaseGraphStorage):
    @staticmethod
    def load_nx_graph(file_name):
        print("no preloading of graph with neo4j in production")

    def __init__(self, namespace, global_config, embedding_func):
        super().__init__(
            namespace=namespace,
            global_config=global_config,
            embedding_func=embedding_func,
        )
        self._driver = None
        self._driver_lock = asyncio.Lock()
        URI = os.environ["NEO4J_URI"]
        USERNAME = os.environ["NEO4J_USERNAME"]
        PASSWORD = os.environ["NEO4J_PASSWORD"]
        self._driver: AsyncDriver = AsyncGraphDatabase.driver(URI, auth=(USERNAME, PASSWORD))
        return None

    def __post_init__(self):
        self._node_embed_algorithms = {
            "node2vec": self._node2vec_embed,
        }

    async def close(self):
        if self._driver:
            await self._driver.close()
            self._driver = None

    async def __aexit__(self, exc_type, exc, tb):
        if self._driver:
            await self._driver.close()

    async def index_done_callback(self):
        print("KG successfully indexed.")

    async def has_node(self, node_id: str) -> bool:
        entity_name_label = node_id.strip('"')

        async with self.driver.session() as session:
            query = (
                f"MATCH (n:`{entity_name_label}`) RETURN count(n) > 0 AS node_exists"
            )
            result = await session.run(query)
            single_result = await result.single()
            logger.debug(
                f'{inspect.currentframe().f_code.co_name}:query:{query}:result:{single_result["node_exists"]}'
            )
            return single_result["node_exists"]

    async def has_edge(self, source_node_id: str, target_node_id: str) -> bool:
        entity_name_label_source = source_node_id.strip('"')
        entity_name_label_target = target_node_id.strip('"')

        async with self.driver.session() as session:
            query = (
                f"MATCH (a:`{entity_name_label_source}`)-[r]-(b:`{entity_name_label_target}`) "
                "RETURN COUNT(r) > 0 AS edgeExists"
            )
            result = await session.run(query)
            single_result = await result.single()
            logger.debug(
                f'{inspect.currentframe().f_code.co_name}:query:{query}:result:{single_result["edgeExists"]}'
            )
            return single_result["edgeExists"]

    async def get_node(self, node_id: str) -> Union[dict, None]:
        async with self.driver.session() as session:
            entity_name_label = node_id.strip('"')
            query = f"MATCH (n:`{entity_name_label}`) RETURN n"
            result = await session.run(query)
            record = await result.single()
            if record:
                node = record["n"]
                node_dict = dict(node)
                logger.debug(
                    f"{inspect.currentframe().f_code.co_name}: query: {query}, result: {node_dict}"
                )
                return node_dict
            return None

    async def node_degree(self, node_id: str) -> int:
        entity_name_label = node_id.strip('"')

        async with self.driver.session() as session:
            query = f"""
                MATCH (n:`{entity_name_label}`)
                RETURN COUNT{{ (n)--() }} AS totalEdgeCount
            """
            result = await session.run(query)
            record = await result.single()
            if record:
                edge_count = record["totalEdgeCount"]
                logger.debug(
                    f"{inspect.currentframe().f_code.co_name}:query:{query}:result:{edge_count}"
                )
                return edge_count
            else:
                return None

    async def edge_degree(self, src_id: str, tgt_id: str) -> int:
        entity_name_label_source = src_id.strip('"')
        entity_name_label_target = tgt_id.strip('"')
        src_degree = await self.node_degree(entity_name_label_source)
        trg_degree = await self.node_degree(entity_name_label_target)

        # Convert None to 0 for addition
        src_degree = 0 if src_degree is None else src_degree
        trg_degree = 0 if trg_degree is None else trg_degree

        degrees = int(src_degree) + int(trg_degree)
        logger.debug(
            f"{inspect.currentframe().f_code.co_name}:query:src_Degree+trg_degree:result:{degrees}"
        )
        return degrees

    async def get_edge(
        self, source_node_id: str, target_node_id: str
    ) -> Union[dict, None]:
        entity_name_label_source = source_node_id.strip('"')
        entity_name_label_target = target_node_id.strip('"')
        """
        Find all edges between nodes of two given labels

        Args:
            source_node_label (str): Label of the source nodes
            target_node_label (str): Label of the target nodes

        Returns:
            list: List of all relationships/edges found
        """
        async with self.driver.session() as session:
            query = f"""
            MATCH (start:`{entity_name_label_source}`)-[r]->(end:`{entity_name_label_target}`)
            RETURN properties(r) as edge_properties
            LIMIT 1
            """.format(
                entity_name_label_source=entity_name_label_source,
                entity_name_label_target=entity_name_label_target,
            )

            result = await session.run(query)
            record = await result.single()
            if record:
                result = dict(record["edge_properties"])
                logger.debug(
                    f"{inspect.currentframe().f_code.co_name}:query:{query}:result:{result}"
                )
                return result
            else:
                return None

    async def get_node_edges(self, source_node_id: str) -> List[Tuple[str, str]]:
        node_label = source_node_id.strip('"')

        """
        Retrieves all edges (relationships) for a particular node identified by its label.
        :return: List of dictionaries containing edge information
        """
        query = f"""MATCH (n:`{node_label}`)
                OPTIONAL MATCH (n)-[r]-(connected)
                RETURN n, r, connected"""
        async with self.driver.session() as session:
            results = await session.run(query)
            edges = []
            async for record in results:
                source_node = record["n"]
                connected_node = record["connected"]

                source_label = (
                    list(source_node.labels)[0] if source_node.labels else None
                )
                target_label = (
                    list(connected_node.labels)[0]
                    if connected_node and connected_node.labels
                    else None
                )

                if source_label and target_label:
                    edges.append((source_label, target_label))

            return edges

    RELATION_TYPE_MAPPING = {
        # Structure : (source_type, target_type) : new_label
        ('activity', 'positive_point'): 'HAS_FEATURE',
        ('positive_point', 'activity'): 'HAS_FEATURE',
        ('activity', 'negative_point'): 'HAS_FEATURE',
        ('negative_point', 'activity'): 'HAS_FEATURE',
        ('activity', 'recommandation'): 'RECOMMENDS',
        ('recommandation', 'activity'): 'RECOMMENDS',
        ('user', 'user_preference'): 'LIKES',
        ('user_preference', 'user'): 'LIKES',
        ('user', 'user_attribute'): 'HAS_INFORMATION',
        ('user_attribute', 'user'): 'HAS_INFORMATION',
        ('event', 'date'): 'OCCURS_ON',
        ('date', 'event'): 'OCCURS_ON',
        ('event', 'positive_point'): 'HAS_FEATURE',
        ('positive_point', 'event'): 'HAS_FEATURE',
        ('event', 'negative_point'): 'HAS_FEATURE',
        ('negative_point', 'event'): 'HAS_FEATURE',
        ('user', 'memo'): 'HAS_MEMO',
        ('memo', 'user'): 'HAS_MEMO',
        ('memo', 'date'): 'OCCURS_ON',
        ('date', 'memo'): 'OCCURS_ON',
        ('memo', 'memo_user'): 'IMPACT_USER',
        ('memo_user', 'memo'): 'IMPACT_USER',     
        ('memo', 'note'): 'HAS_DETAIL',
        ('note', 'memo'): 'HAS_DETAIL',      
        ('memo', 'city'): 'LOCATED_IN',
        ('city', 'memo'): 'LOCATED_IN',    
        ('memo', 'priority'): 'HAS_PRIORITY',
        ('priority', 'memo'): 'HAS_PRIORITY',                         
    }

    @property
    def driver(self):
        return self._driver

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(
            (
                neo4jExceptions.ServiceUnavailable,
                neo4jExceptions.TransientError,
                neo4jExceptions.WriteServiceUnavailable,
                neo4jExceptions.ClientError,
            )
        ),
    )
    async def upsert_node(self, node_id: str, node_data: Dict[str, Any]):
        """
        Upsert a node in the Neo4j database.

        Args:
            node_id: The unique identifier for the node
            node_data: Dictionary of node properties
        """
        # Vérification de la connexion
        if not self._driver:
            logger.error("❌ Connexion Neo4j non initialisée")
            return

        # Log TRÈS détaillé
        logger.debug(f"🔍 DEBUG upsert_node - node_id: {node_id}")
        logger.debug(f"🔍 DEBUG upsert_node - node_data BRUT: {node_data}")
        logger.debug(f"🔍 DEBUG upsert_node - node_data keys: {list(node_data.keys())}")
        logger.debug(f"🔍 DEBUG upsert_node - node_data types: {[type(val) for val in node_data.values()]}")

        # Validation des propriétés
        if "custom_id" in node_data:
            logger.debug(f"🏷️ Custom ID trouvé pour le nœud {node_id}: {node_data['custom_id']}")

        # Vérifier que toutes les propriétés sont des types supportés par Neo4j
        for key, value in list(node_data.items()):
            if not isinstance(value, (str, int, float, bool, list)):
                logger.warning(f"⚠️ Propriété {key} de type {type(value)} non supportée par Neo4j, conversion en str")
                node_data[key] = str(value)


        label = node_id.strip('"')
        logger.debug(f"🏷️ Label du nœud : {label}")

        properties = node_data

        async def _do_upsert(tx: AsyncManagedTransaction):
            try:
                # Convertir toutes les propriétés en types supportés par Neo4j
                clean_properties = {}
                for key, value in properties.items():
                    if isinstance(value, (str, int, float, bool)):
                        clean_properties[key] = value
                    else:
                        clean_properties[key] = str(value)

                # Log détaillé des propriétés
                logger.debug(f"🧹 clean_properties avant insertion: {clean_properties}")
                logger.debug(f"🧹 clean_properties keys: {list(clean_properties.keys())}")

                query = f"""
                MERGE (n:`{label}`)
                SET n = $properties
                RETURN n
                """
                result = await tx.run(query, properties=clean_properties)
                record = await result.single()

                if record:
                    logger.debug(f"✅ Nœud créé/mis à jour avec succès : {label}")
                    # Log du nœud inséré
                    node_record = record.data()['n']
                    logger.debug(f"🔬 DEBUG nœud inséré : {node_record}")
                else:
                    logger.warning(f"⚠️ Aucun nœud créé pour : {label}")
            except Exception as e:
                logger.error(f"❌ Erreur lors de la création du nœud : {e}")
                raise

        try:
            async with self.driver.session() as session:
                await session.execute_write(_do_upsert)
        except Exception as e:
            logger.error(f"❌ Erreur lors de l'exécution de la transaction : {e}")
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(
            (
                neo4jExceptions.ServiceUnavailable,
                neo4jExceptions.TransientError,
                neo4jExceptions.WriteServiceUnavailable,
            )
        ),
    )
    async def upsert_edge(
        self, source_node_id: str, target_node_id: str, edge_data: Dict[str, Any]
    ):
        """
        Upsert an edge and its properties between two nodes identified by their labels.

        Args:
            source_node_id (str): Label of the source node (used as identifier)
            target_node_id (str): Label of the target node (used as identifier)
            edge_data (dict): Dictionary of properties to set on the edge
        """
        source_node_label = source_node_id.strip('"')
        target_node_label = target_node_id.strip('"')
        edge_properties = edge_data

        async def _do_upsert_edge(tx: AsyncManagedTransaction):
            # Récupération des données des nœuds source et cible
            source_node_data = await tx.run(
                "MATCH (n {custom_id: $source_node_id}) RETURN n", 
                {"source_node_id": source_node_id}
            )
            source_node_data = await source_node_data.single()
            source_node_data = dict(source_node_data["n"])

            target_node_data = await tx.run(
                "MATCH (n {custom_id: $target_node_id}) RETURN n", 
                {"target_node_id": target_node_id}
            )
            target_node_data = await target_node_data.single()
            target_node_data = dict(target_node_data["n"])

            # Détermination du label de relation
            relation_key = (source_node_data.get('entity_type'), target_node_data.get('entity_type'))
            reverse_relation_key = (target_node_data.get('entity_type'), source_node_data.get('entity_type'))

            # Vérification stricte du sens de la relation
            if relation_key in self.RELATION_TYPE_MAPPING:
                new_label = self.RELATION_TYPE_MAPPING[relation_key]
                create_relation_query = """
                MATCH (source {custom_id: $source_node_id})
                MATCH (target {custom_id: $target_node_id})
                MERGE (source)-[r:$new_label]->(target)
                SET r += $edge_data
                RETURN r
                """
            elif reverse_relation_key in self.RELATION_TYPE_MAPPING:
                # Si la relation inverse existe dans le mapping, on lève une erreur
                logger.warning(f"Tentative de créer une relation dans le mauvais sens : {relation_key}")
                return None
            else:
                # Relation par défaut si non définie
                new_label = 'DIRECTED'
                create_relation_query = """
                MATCH (source {custom_id: $source_node_id})
                MATCH (target {custom_id: $target_node_id})
                MERGE (source)-[r:$new_label]->(target)
                SET r += $edge_data
                RETURN r
                """

            # Exécution de la requête
            result = await tx.run(create_relation_query, {
                "source_node_id": source_node_id,
                "target_node_id": target_node_id,
                "new_label": new_label,
                "edge_data": edge_data
            })

            return await result.single()

        try:
            async with self.driver.session() as session:
                await session.execute_write(_do_upsert_edge)
        except Exception as e:
            logger.error(f"Error during edge upsert: {str(e)}")
            raise

    async def _node2vec_embed(self):
        print("Implemented but never called.")

    async def categorize_activities(
        self, 
        activity_categories_manager, 
        use_model_func=None, 
        session=None
    ) -> Dict[str, int]:
        """
        Catégorise les activités dans la base de données Neo4j.
        
        Args:
            activity_categories_manager: Gestionnaire des catégories d'activités
            use_model_func: Fonction optionnelle pour générer des catégories via LLM
            session: Session Neo4j optionnelle
        
        Returns:
            Dictionnaire avec les compteurs de catégorisation
        """
        # Relation type mapping
        RELATION_TYPE_MAPPING = {
            ('activity', 'ActivityCategory'): 'CLASSIFIED_AS',
        }
        
        # Utiliser la session existante ou en créer une nouvelle
        if session is None:
            session = self.driver.session()
        
        async with session:
            # Initialiser les catégories prédéfinies
            init_query = """
            MERGE (restauration:ActivityCategory {name: 'Restauration'})
            MERGE (culture:ActivityCategory {name: 'Culture et Loisirs'})
            MERGE (sport:ActivityCategory {name: 'Sport et Fitness'})
            MERGE (voyage:ActivityCategory {name: 'Voyage et Tourisme'})
            MERGE (formation:ActivityCategory {name: 'Formation et Éducation'})
            MERGE (bienetre:ActivityCategory {name: 'Bien-être et Santé'})
            MERGE (pro:ActivityCategory {name: 'Événements Professionnels'})
            MERGE (unknown:ActivityCategory {name: 'Unknown'})
            
            // Requête pour récupérer les activités sans catégorie
            WITH 1 as dummy
            MATCH (n {entity_type: 'activity'})
            WHERE NOT (n)-[:CLASSIFIED_AS]->(:ActivityCategory)
            RETURN n.description as description, elementId(n) as node_id, labels(n) as node_labels
            """
            
            # Exécuter l'initialisation et récupérer les activités
            result = await session.run(init_query)
            activities = await result.data()
            
            # Compteurs de catégorisation
            categorization_counts = {
                'total': 0,
                'categorized': 0,
                'uncategorized': 0
            }
            
            for activity in activities:
                description = activity['description']
                node_id = activity['node_id']
                node_labels = activity['node_labels']
                
                # Utiliser le gestionnaire de catégories pour déterminer la catégorie
                if use_model_func:
                    category = await use_model_func(description)
                else:
                    category = activity_categories_manager.get_category(description)
                
                # Catégorisation par défaut si aucune catégorie n'est trouvée
                if not category:
                    category = 'Unknown'
                    categorization_counts['uncategorized'] += 1
                else:
                    categorization_counts['categorized'] += 1
                
                categorization_counts['total'] += 1
                
                # Requête pour créer la relation de catégorisation
                categorize_query = """
                MATCH (activity) WHERE elementId(activity) = $node_id
                WITH activity
                MATCH (cat:ActivityCategory {name: $category_name})
                MERGE (activity)-[r:CLASSIFIED_AS]->(cat)
                RETURN activity, cat
                """
                
                try:
                    await session.run(
                        categorize_query, 
                        node_id=node_id, 
                        category_name=category
                    )
                    logger.debug(f"🏷️ Catégorisation de l'activité {node_id} dans la catégorie {category}")
                except Exception as e:
                    logger.error(f"❌ Erreur lors de la catégorisation de l'activité {node_id} : {e}")
            
            logger.debug("📊 Résumé de la catégorisation :")
            logger.debug(f"   - Total d'activités : {categorization_counts['total']}")
            logger.debug(f"   - Activités catégorisées : {categorization_counts['categorized']}")
            logger.debug(f"   - Activités non catégorisées : {categorization_counts['uncategorized']}")
            
            return categorization_counts

    async def categorize_cities(
        self,
        custom_id: str,
        city_name: str,
    ) -> Dict[str, Any]:
        """
        Associe une activité à une ville.
        Si le nœud ville n'existe pas, il sera créé.

        Args:
            custom_id (str): Le custom_id de l'activité
            city_name (str): Le nom de la ville

        Returns:
            Dict[str, Any]: Les informations de l'activité et de la ville
        """
        query = """
        MATCH (activity {custom_id: $custom_id})
        WITH activity
        
        // Vérifier s'il existe déjà une relation LOCATED_IN
        OPTIONAL MATCH (activity)-[existing_relation:LOCATED_IN]->(existing_city:City)
        
        // Créer ou récupérer la nouvelle ville
        MERGE (city:City {name: $city_name})
        ON CREATE SET city.entity_type = 'city'
        
        WITH activity, city, existing_relation, existing_city
        
        // Gérer la relation
        FOREACH (_ IN CASE 
            WHEN existing_relation IS NULL THEN [1]
            WHEN existing_city.name <> $city_name THEN [1]
            ELSE []
        END |
            MERGE (activity)-[:LOCATED_IN]->(city)
            SET activity.city_conflict = CASE WHEN existing_city.name <> $city_name THEN true ELSE false END
        )
        
        RETURN activity, city, 
               CASE 
                   WHEN activity.city_conflict IS NOT NULL AND activity.city_conflict = true THEN 'conflict' 
                   WHEN existing_relation IS NULL THEN 'created' 
                   ELSE 'existing' 
               END AS city_status
        """
        
        async with self.driver.session() as session:
            result = await session.run(
                query,
                {
                    "custom_id": custom_id,
                    "city_name": city_name
                }
            )
            
            records = await result.data()
            
            if records:
                logger.info(f"Résultats de la requête : {records}")
                record = records[0]
            else:
                logger.warning("Aucun résultat trouvé pour la requête")
                return None
            
            activity = record["activity"]
            city = record["city"]
            city_status = record["city_status"]
            
            if city_status == 'conflict':
                logger.warning(f"Conflit de ville détecté pour l'activité {custom_id}. Ancienne ville différente de {city_name}")
            
            logger.debug(f"City {city_name}: {city_status}")
            
            return {
                "activity": dict(activity),
                "city": dict(city),
                "city_status": city_status
            }

    async def categorize_dates(
        self, 
        custom_id: str, 
        date_label: str
    ):
        """
        Associe un événement à un nœud de date dans le graphe de connaissances.
        
        Args:
            custom_id (str): Identifiant personnalisé de l'événement
            date_label (str): Étiquette de la date (format YYYY-MM-DD)
        """
        async with self.driver.session() as session:
            try:
                # Requête pour créer ou récupérer le nœud de date et le lier à l'événement
                query = """
                MATCH (event {custom_id: $custom_id})
                WITH event
                
                // Vérifier s'il existe déjà une relation OCCURS_ON
                OPTIONAL MATCH (event)-[existing_relation:OCCURS_ON]->(existing_date:Date)
                
                // Créer ou récupérer la nouvelle date
                MERGE (date:Date {label: $date_label})
                ON CREATE SET date.entity_type = 'date'

                WITH event, date, existing_relation, existing_date
                
                // Gérer la relation
                FOREACH (_ IN CASE 
                    WHEN existing_relation IS NULL THEN [1]
                    WHEN existing_date.label <> $date_label THEN [1]
                    ELSE []
                END |
                    MERGE (event)-[:OCCURS_ON]->(date)
                    SET event.date_conflict = CASE WHEN existing_date.label <> $date_label THEN true ELSE false END
                )
                
                RETURN event, date, 
                       CASE 
                           WHEN event.date_conflict IS NOT NULL AND event.date_conflict = true THEN 'conflict' 
                           WHEN existing_relation IS NULL THEN 'created' 
                           ELSE 'existing' 
                       END AS date_status
                """
                
                # Exécuter l'initialisation et récupérer les activités
                result = await session.run(
                    query,
                    {
                        "custom_id": custom_id,
                        "date_label": date_label
                    }
                )
                
                records = await result.data()
                
                if records:
                    logger.info(f"Résultats de la requête : {records}")
                    record = records[0]
                else:
                    logger.warning("Aucun résultat trouvé pour la requête")
                    return None
                
                event = record["event"]
                date = record["date"]
                date_status = record["date_status"]
                
                if date_status == 'conflict':
                    logger.warning(f"Conflit de date détecté pour l'événement {custom_id}. Ancienne date différente de {date_label}")
                
                logger.debug(f"Date {date_label}: {date_status}")
                
                return {
                    "event": dict(event),
                    "date": dict(date),
                    "date_status": date_status
                }
            
            except Exception as e:
                logger.error(f"❌ Erreur lors de l'association de la date : {e}")
                return None

    async def categorize_memos(
        self, 
        custom_id: str,  # Représente l'identifiant du mémo
        user_id: str,  # Représente l'identifiant obligatoire de l'utilisateur
    ):
        """
        Crée une relation entre un mémo et un utilisateur dans le graphe de connaissances.
        
        Args:
            custom_id (str): Identifiant personnalisé du mémo.
            user_id (str): Identifiant personnalisé de l'utilisateur.
        
        Returns:
            Optional[Dict]: Résultats de l'association du mémo à l'utilisateur, ou None si échec.
        """
        async with self.driver.session() as session:
            try:
                # Log de débogage
                logger.info(f"Requête Cypher avec custom_id: {custom_id}, user_id: {user_id}")
                
                # Vérifier l'existence des nœuds
                check_memo_query = "MATCH (memo {custom_id: $custom_id, entity_type: 'memo'}) RETURN memo"
                check_user_query = """
                MATCH (user) 
                WHERE user.entity_type = 'user' AND 
                (
                    user.custom_id = $user_id OR 
                    user.custom_id = $normalized_user_id
                )
                RETURN user
                """
                
                # Normaliser l'ID utilisateur
                normalized_user_id = self.normalize_label(user_id) if user_id else None
                
                memo_result = await session.run(check_memo_query, {"custom_id": custom_id})
                user_result = await session.run(check_user_query, {
                    "user_id": user_id, 
                    "normalized_user_id": normalized_user_id
                })
                
                memo_records = await memo_result.data()
                user_records = await user_result.data()
                
                logger.info(f"Mémos trouvés : {memo_records}")
                logger.info(f"Utilisateurs trouvés : {user_records}")
                
                # Exécution de la requête Cypher
                query = """
                MATCH (memo {custom_id: $custom_id})
                OPTIONAL MATCH (user {
                    custom_id: $normalized_user_id, 
                    entity_type: 'user'
                })
                OPTIONAL MATCH (user)-[existing_relation:HAS_MEMO]->(memo)
                WITH memo, user, existing_relation
                WHERE user IS NOT NULL AND existing_relation IS NULL
                CREATE (user)-[:HAS_MEMO]->(memo)
                RETURN memo, 
                    user, 
                    CASE 
                        WHEN user IS NOT NULL AND existing_relation IS NULL THEN 'created'
                        WHEN user IS NOT NULL AND existing_relation IS NOT NULL THEN 'existing'
                        ELSE 'no_user'
                    END AS memo_status
                """
                
                # Log de débogage
                logger.info(f"Requête Cypher avec custom_id: {custom_id}, user_id: {user_id}, normalized_user_id: {normalized_user_id}")
                
                # Exécution de la requête Cypher
                result = await session.run(query, {
                    "custom_id": custom_id, 
                    "user_id": user_id,
                    "normalized_user_id": normalized_user_id
                })
                records = await result.data()
                
                # Extraction du résultat
                if records:
                    logger.info(f"Résultats de la requête : {records}")
                    record = records[0]
                else:
                    logger.warning("Aucun résultat trouvé pour la requête")
                    return None

                # Récupération des données
                memo = record["memo"]
                user = record["user"]
                memo_status = record["memo_status"]

                logger.info(f"Mémo {custom_id} : Relation {memo_status}")
                
                return {
                    "memo": dict(memo) if memo else None,
                    "user": dict(user) if user else None,
                    "memo_status": memo_status
                }
            
            except Exception as e:
                logger.error(f"❌ Erreur lors de l'association du mémo '{custom_id}' avec l'utilisateur '{user_id}' : {e}")
                return None


    async def merge_duplicate_users(self):
        """
        Fusionne les nœuds utilisateurs qui ont le même custom_id.
        Conserve toutes les relations existantes.
        """
        if not self._driver:
            logger.error("❌ Connexion Neo4j non initialisée")
            return

        async def _do_merge(tx):
            # Trouver les custom_id qui ont des doublons
            find_duplicates_query = """
            MATCH (u:user)
            WHERE u.custom_id IS NOT NULL
            WITH u.custom_id as cid, collect(u) as users
            WHERE size(users) > 1
            RETURN cid, users
            """
            
            result = await tx.run(find_duplicates_query)
            records = await result.records()  # Utiliser records() au lieu de fetch()
            
            for record in records:
                custom_id = record["cid"]
                users = record["users"]
                logger.debug(f"Fusion des utilisateurs avec custom_id: {custom_id}")
                
                # Garder le premier nœud et fusionner les autres
                primary_user = users[0]
                duplicate_users = users[1:]
                
                for duplicate in duplicate_users:
                    # Transférer toutes les relations entrantes
                    merge_in_query = """
                    MATCH (duplicate) WHERE id(duplicate) = $duplicate_id
                    MATCH (primary) WHERE id(primary) = $primary_id
                    MATCH (source)-[r]->(duplicate)
                    WHERE NOT source = primary
                    CALL apoc.merge.relationship(source, type(r), r.properties, primary) YIELD rel
                    DELETE r
                    """
                    
                    # Transférer toutes les relations sortantes
                    merge_out_query = """
                    MATCH (duplicate) WHERE id(duplicate) = $duplicate_id
                    MATCH (primary) WHERE id(primary) = $primary_id
                    MATCH (duplicate)-[r]->(target)
                    WHERE NOT target = primary
                    CALL apoc.merge.relationship(primary, type(r), r.properties, target) YIELD rel
                    DELETE r
                    """
                    
                    # Supprimer le nœud dupliqué
                    delete_query = """
                    MATCH (n) WHERE id(n) = $node_id
                    DETACH DELETE n
                    """
                    
                    params = {
                        "duplicate_id": duplicate.id,
                        "primary_id": primary_user.id,
                        "node_id": duplicate.id
                    }
                    
                    try:
                        await tx.run(merge_in_query, params)
                        await tx.run(merge_out_query, params)
                        await tx.run(delete_query, params)
                        logger.debug(f"✅ Nœud fusionné et supprimé: {duplicate.id}")
                    except Exception as e:
                        logger.error(f"❌ Erreur lors de la fusion du nœud {duplicate.id}: {str(e)}")

        try:
            async with self.driver.session() as session:
                await session.execute_write(_do_merge)
                logger.debug("✅ Fusion des utilisateurs terminée")
        except Exception as e:
            logger.error(f"❌ Erreur lors de la fusion des utilisateurs: {str(e)}")


    async def extract_subgraph(self, custom_ids):
        """
        Extrait un sous-graphe pour une liste de custom_ids
        
        Returns:
            dict: Un dictionnaire structuré similaire à chunk_entity_relation_graph
        """
        async with self.driver.session() as session:
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
            chunk_entity_relation_graph = {
                "text_chunks": {},
                "entities": {},
                "relations": []
            }
            
            async for record in result:
                # Traitement du nœud source
                source_id = record["source_node"].element_id
                source_properties = dict(record["source_node"])
                
                if source_id not in chunk_entity_relation_graph["entities"]:
                    # Vérifier et extraire l'entity_id spécifique
                    entity_id = source_properties.get('entity_id')
                    if not (isinstance(entity_id, str) and entity_id.startswith('ent-')):
                        entity_id = None
                    
                    chunk_entity_relation_graph["entities"][source_id] = {
                        "id": source_id,
                        "labels": record["source_labels"],
                        "properties": source_properties,
                        "entity_id": entity_id or source_properties.get('custom_id', source_id)
                    }
                
                # Traitement du nœud cible
                target_id = record["target_node"].element_id
                target_properties = dict(record["target_node"])
                
                if target_id not in chunk_entity_relation_graph["entities"]:
                    # Vérifier et extraire l'entity_id spécifique
                    entity_id = target_properties.get('entity_id')
                    if not (isinstance(entity_id, str) and entity_id.startswith('ent-')):
                        entity_id = None
                    
                    chunk_entity_relation_graph["entities"][target_id] = {
                        "id": target_id,
                        "labels": record["target_labels"],
                        "properties": target_properties,
                        "entity_id": entity_id or target_properties.get('custom_id', target_id)
                    }
                
                # Ajout de la relation
                chunk_entity_relation_graph["relations"].append({
                    "source_id": source_id,
                    "target_id": target_id,
                    "type": record["relationship"].type,
                    "id": record["relationship"].element_id,  # Ajout de l'ID de relation
                    "relation_id": record["relationship"]["relation_id"],  # Récupération de l'attribut personnalisé relation_id
                    "properties": dict(record["relationship"])
                })
            
            # Conversion des entités en liste
            chunk_entity_relation_graph["entities"] = list(chunk_entity_relation_graph["entities"].values())
            
            logger.debug(f"Sous-graphe extrait pour {len(custom_ids)} nœuds")
            return chunk_entity_relation_graph

    async def afilter_nodes(self, node_ids):
        """
        Filtre les nœuds du graphe Neo4j en utilisant custom_id de manière asynchrone
        
        Args:
            node_ids (List[str]): Liste des custom_id de nœuds à filtrer
        
        Returns:
            list: Liste des nœuds et relations filtrés
        """
        async with self.driver.session() as session:
            query = """
            // Trouver les nœuds par leur custom_id
            MATCH (n)
            WHERE n.custom_id IN $node_ids
            
            // Récupérer les nœuds et leurs relations
            MATCH (n)-[r]-(connected)
            RETURN 
                n AS source_node, 
                r AS relationship, 
                connected AS target_node,
                labels(n) AS source_labels,
                labels(connected) AS target_labels,
                properties(n) AS source_properties,
                properties(r) AS relationship_properties,
                properties(connected) AS target_properties
            """
            
            try:
                logger.debug(f"afilter_nodes - Valeur de node_ids : {node_ids}")
                
                result = await session.run(query, {"node_ids": node_ids})
                
                # Collecter les résultats
                filtered_results = []
                async for record in result:
                    filtered_results.append({
                        'source_node': {
                            'node': dict(record['source_node']),
                            'labels': record['source_labels'],
                            'properties': record['source_properties']
                        },
                        'relationship': {
                            'relation': dict(record['relationship']),
                            'properties': record['relationship_properties']
                        },
                        'target_node': {
                            'node': dict(record['target_node']),
                            'labels': record['target_labels'],
                            'properties': record['target_properties']
                        }
                    })
                
                logger.debug(f"Filtrage réussi : {len(filtered_results)} résultats trouvés")
                
                return filtered_results
            
            except Exception as e:
                logger.error(f"Erreur lors du filtrage du graphe Neo4j : {e}")
                raise

    async def aextract_filtered_ids(self, filtered_results):
        """
        Extrait les entity_ids et relation_ids à partir des résultats filtrés de manière asynchrone.
        
        Args:
            filtered_results (list): Liste des résultats filtrés
        
        Returns:
            dict: Dictionnaire contenant les node_ids et relation_ids
        """
        filtered_ids = {
            'node_ids': set(),
            'relation_ids': set()
        }
        
        for result in filtered_results:
            # Extraire l'entity_id du nœud source
            source_entity_id = result['source_node']['properties'].get('entity_id')
            if source_entity_id:
                filtered_ids['node_ids'].add(source_entity_id)
            
            # Extraire l'entity_id du nœud cible
            target_entity_id = result['target_node']['properties'].get('entity_id')
            if target_entity_id:
                filtered_ids['node_ids'].add(target_entity_id)
            
            # Extraire le relation_id
            relation_id = result['relationship']['properties'].get('relation_id')
            if relation_id:
                filtered_ids['relation_ids'].add(relation_id)
        
        # Convertir les sets en listes
        filtered_ids['node_ids'] = list(filtered_ids['node_ids'])
        filtered_ids['relation_ids'] = list(filtered_ids['relation_ids'])
        
        return filtered_ids

    async def get_filtered_ids(self, vdb_filter):
        
        logger.debug(f"get_filtered_ids - Type de vdb_filter : {type(vdb_filter)}")
        logger.debug(f"get_filtered_ids - Contenu de vdb_filter : {vdb_filter}")

        if isinstance(vdb_filter, dict):
            logger.debug(f"get_filtered_ids - Clés de vdb_filter : {vdb_filter.keys()}")
            for key, value in vdb_filter.items():
                logger.debug(f"get_filtered_ids - Clé {key} : {type(value)}, Valeur : {value}")
        
        # Utiliser la méthode filter_nodes de la classe courante
        filtered_results = await self.afilter_nodes(vdb_filter)
        
        # Extraire les IDs en utilisant la méthode extract_filtered_ids de la classe courante
        filtered_ids = await self.aextract_filtered_ids(filtered_results)
        
        # Afficher les informations de base
        #logger.info(f"Type du Resultat du filtrage: {type(filtered_results)}")
        #logger.info(f"Resultats du filtrage: {filtered_results}")
        #logger.info(f"IDs collectés : {filtered_ids}")
        
        return filtered_ids

    def normalize_label(self, label: str) -> str:
        """
        Normalise un label en supprimant les espaces et en convertissant en minuscules.
        
        Args:
            label (str): Le label à normaliser
        
        Returns:
            str: Le label normalisé
        """
        return label.replace(" ", "").lower()
