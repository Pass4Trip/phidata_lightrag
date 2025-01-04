import asyncio
import json
import re
from typing import Optional, Dict, Any
from tqdm.asyncio import tqdm as tqdm_async
from typing import Union
from collections import Counter, defaultdict
import warnings
from .utils import (
    logger,
    clean_str,
    compute_mdhash_id,
    decode_tokens_by_tiktoken,
    encode_string_by_tiktoken,
    is_float_regex,
    list_of_list_to_csv,
    pack_user_ass_to_openai_messages,
    split_string_by_multi_markers,
    truncate_list_by_token_size,
    process_combine_contexts,
    compute_args_hash,
    handle_cache,
    save_to_cache,
    CacheData,
)
from .base import (
    BaseGraphStorage,
    BaseKVStorage,
    BaseVectorStorage,
    TextChunkSchema,
    QueryParam,
)
from .prompt import GRAPH_FIELD_SEP, PROMPTS


def chunking_by_token_size(
    content: str, overlap_token_size=128, max_token_size=1024, tiktoken_model="gpt-4o"
):
    tokens = encode_string_by_tiktoken(content, model_name=tiktoken_model)
    results = []
    for index, start in enumerate(
        range(0, len(tokens), max_token_size - overlap_token_size)
    ):
        chunk_content = decode_tokens_by_tiktoken(
            tokens[start : start + max_token_size], model_name=tiktoken_model
        )
        results.append(
            {
                "tokens": min(max_token_size, len(tokens) - start),
                "content": chunk_content.strip(),
                "chunk_order_index": index,
            }
        )
    return results


async def _handle_entity_relation_summary(
    entity_or_relation_name: str,
    description: str,
    global_config: dict,
) -> str:
    use_llm_func: callable = global_config["llm_model_func"]
    llm_max_tokens = global_config["llm_model_max_token_size"]
    tiktoken_model_name = global_config["tiktoken_model_name"]
    summary_max_tokens = global_config["entity_summary_to_max_tokens"]
    language = global_config["addon_params"].get(
        "language", PROMPTS["DEFAULT_LANGUAGE"]
    )

    tokens = encode_string_by_tiktoken(description, model_name=tiktoken_model_name)
    if len(tokens) < summary_max_tokens:  # No need for summary
        return description
    prompt_template = PROMPTS["summarize_entity_descriptions"]
    use_description = decode_tokens_by_tiktoken(
        tokens[:llm_max_tokens], model_name=tiktoken_model_name
    )
    context_base = dict(
        entity_name=entity_or_relation_name,
        description_list=use_description.split(GRAPH_FIELD_SEP),
        language=language,
    )
    use_prompt = prompt_template.format(**context_base)
    logger.debug(f"Trigger summary: {entity_or_relation_name}")
    summary = await use_llm_func(use_prompt, max_tokens=summary_max_tokens)
    return summary


async def _handle_single_entity_extraction(
    record_attributes: list[str],
    chunk_key: str,
):
    if len(record_attributes) < 4:
        logger.warning(f"Not enough record attributes: {len(record_attributes)}")
        return None
    
    # More robust check for entity type
    first_attr = record_attributes[0]
    
    if first_attr.lower() != "entity":
        logger.warning(f"First attribute is not 'entity': {first_attr}")
        return None
    
    # add this record as a node in the G
    entity_name = clean_str(record_attributes[1])
    if not entity_name.strip():
        logger.warning("Entity name is empty after cleaning")
        return None
    
    entity_type = clean_str(record_attributes[2])
    entity_description = clean_str(record_attributes[3])
    entity_source_id = chunk_key
    
    result = dict(
        entity_name=entity_name,
        entity_type=entity_type,
        description=entity_description,
        source_id=entity_source_id,
    )
    
    logger.debug(f"DEBUG: Returning entity: {result}")
    return result


async def _handle_single_relationship_extraction(
    record_attributes: list[str],
    chunk_key: str,
):
    if len(record_attributes) < 5:
        logger.warning(f"Not enough record attributes: {len(record_attributes)}")
        return None
    
    first_attr = record_attributes[0]
    if first_attr.lower() != "relationship":
        logger.warning(f"First attribute is not 'relationship': {first_attr}")
        return None
    
    src_id = clean_str(record_attributes[1])
    tgt_id = clean_str(record_attributes[2])
    
    if not src_id or not tgt_id:
        logger.warning(f"Source or target ID is empty: src_id={src_id}, tgt_id={tgt_id}")
        return None
    
    description = clean_str(record_attributes[3])
    keywords = clean_str(record_attributes[4])
    weight = 1
    
    # if len(record_attributes) > 5:
    #     try:
    #         weight = int(record_attributes[5])
    #     except ValueError:
    #         logger.warning(f"Could not convert weight to integer: {record_attributes[5]}")

    weight = (
        float(record_attributes[-1]) if is_float_regex(record_attributes[-1]) else 1.0
    )
    
    result = dict(
        src_id=src_id,
        tgt_id=tgt_id,
        description=description,
        keywords=keywords,
        weight=weight,
        source_id=chunk_key,
    )
    
    logger.debug(f"DEBUG: Returning relationship: {result}")
    return result


async def _merge_nodes_then_upsert(
    entity_name: str,
    nodes_data: list[dict],
    knowledge_graph_inst: BaseGraphStorage,
    global_config: dict,
):
    """Fusionne et met à jour les nœuds dans le graphe de connaissances"""
    try:
        # S'assurer d'utiliser la même boucle d'événements
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
        # Vérifier si le nœud existe déjà
        already_node = await knowledge_graph_inst.get_node(entity_name)
        if already_node is not None:
            already_entitiy_types = [already_node["entity_type"]]
            already_source_ids = split_string_by_multi_markers(already_node["source_id"], [GRAPH_FIELD_SEP])
            already_description = [already_node["description"]]
            # Récupérer les metadata existantes
            already_metadata = {k: v for k, v in already_node.items() 
                              if k not in ["entity_type", "description", "source_id", "entity_name"]}
        else:
            already_entitiy_types = []
            already_source_ids = []
            already_description = []
            already_metadata = {}

        # Fusionner les metadata de tous les nœuds
        metadata = {}
        for node in nodes_data:
            node_metadata = {k: v for k, v in node.items() 
                           if k not in ["entity_type", "description", "source_id", "entity_name"]}
            metadata.update(node_metadata)
        
        # Combiner avec les metadata existantes
        metadata.update(already_metadata)

        entity_type = sorted(
            Counter(
                [dp["entity_type"] for dp in nodes_data] + already_entitiy_types
            ).items(),
            key=lambda x: x[1],
            reverse=True,
        )[0][0]
        description = GRAPH_FIELD_SEP.join(
            sorted(set([dp["description"] for dp in nodes_data] + already_description))
        )
        source_id = GRAPH_FIELD_SEP.join(
            set([dp["source_id"] for dp in nodes_data] + already_source_ids)
        )
        description = await _handle_entity_relation_summary(
            entity_name, description, global_config
        )
        node_data = dict(
            entity_type=entity_type,
            description=description,
            source_id=source_id,
            **metadata  # Ajouter les metadata au nœud
        )
        await knowledge_graph_inst.upsert_node(
            entity_name,
            node_data=node_data,
        )
        node_data["entity_name"] = entity_name
        return node_data
    except Exception as e:
        logger.error(f"Erreur lors de la fusion des nœuds : {e}")
        return None


async def _merge_edges_then_upsert(
    src_id: str,
    tgt_id: str,
    edges_data: list[dict],
    knowledge_graph_inst: BaseGraphStorage,
    global_config: dict,
):
    already_weights = []
    already_source_ids = []
    already_description = []
    already_keywords = []

    if await knowledge_graph_inst.has_edge(src_id, tgt_id):
        already_edge = await knowledge_graph_inst.get_edge(src_id, tgt_id)
        already_weights.append(already_edge["weight"])
        already_source_ids.extend(
            split_string_by_multi_markers(already_edge["source_id"], [GRAPH_FIELD_SEP])
        )
        already_description.append(already_edge["description"])
        already_keywords.extend(
            split_string_by_multi_markers(already_edge["keywords"], [GRAPH_FIELD_SEP])
        )

    weight = sum([dp["weight"] for dp in edges_data] + already_weights)
    description = GRAPH_FIELD_SEP.join(
        sorted(set([dp["description"] for dp in edges_data] + already_description))
    )
    keywords = GRAPH_FIELD_SEP.join(
        sorted(set([dp["keywords"] for dp in edges_data] + already_keywords))
    )
    source_id = GRAPH_FIELD_SEP.join(
        set([dp["source_id"] for dp in edges_data] + already_source_ids)
    )
    for need_insert_id in [src_id, tgt_id]:
        if not (await knowledge_graph_inst.has_node(need_insert_id)):
            await knowledge_graph_inst.upsert_node(
                need_insert_id,
                node_data={
                    "source_id": source_id,
                    "description": description,
                    "entity_type": '"UNKNOWN"',
                },
            )
    description = await _handle_entity_relation_summary(
        f"({src_id}, {tgt_id})", description, global_config
    )
    await knowledge_graph_inst.upsert_edge(
        src_id,
        tgt_id,
        edge_data=dict(
            weight=weight,
            description=description,
            keywords=keywords,
            source_id=source_id,
        ),
    )

    edge_data = dict(
        src_id=src_id,
        tgt_id=tgt_id,
        description=description,
        keywords=keywords,
    )

    return edge_data


async def extract_entities(
    chunks: dict[str, TextChunkSchema],
    knowledge_graph_inst: BaseGraphStorage,
    entity_vdb: BaseVectorStorage,
    relationships_vdb: BaseVectorStorage,
    global_config: dict,
    text_chunks: BaseKVStorage[TextChunkSchema],
    prompt_domain: str = "default",
    metadata: dict = None
) -> Union[BaseGraphStorage, None]:
    """
    Extract entities from text chunks and process them.
    
    Args:
        chunks: Dictionary of text chunks
        knowledge_graph_inst: Graph storage instance
        entity_vdb: Entity vector database
        relationships_vdb: Relationships vector database
        global_config: Global configuration dictionary
        prompt_domain: Prompt domain for extraction
        metadata: Additional metadata
    
    Returns:
        BaseGraphStorage: Updated knowledge graph instance
    """
    # Log d'entrée DÉTAILLÉ
    logger.debug("🚀 DÉBUT de extract_entities")
    logger.debug(f"🔍 Nombre de chunks : {len(chunks)}")
    logger.debug(f"🔍 Domaine du prompt : {prompt_domain}")
    logger.debug(f"🔍 Métadonnées : {metadata}")

    logger.debug(f"Entity extraction using prompt domain: {prompt_domain}")
    use_llm_func: callable = global_config["llm_model_func"]
    entity_extract_max_gleaning = global_config["entity_extract_max_gleaning"]

    ordered_chunks = list(chunks.items())
    # add language and example number params to prompt
    language = global_config["addon_params"].get(
        "language", PROMPTS["DEFAULT_LANGUAGE"]
    )


    domain_ENTITY_TYPES = f"{prompt_domain}_ENTITY_TYPES"
    entity_types = PROMPTS.get(domain_ENTITY_TYPES, PROMPTS[domain_ENTITY_TYPES])  
        
    domain_entity_extraction = f"{prompt_domain}_entity_extraction"
    entity_extract_prompt = PROMPTS.get(domain_entity_extraction, PROMPTS[domain_entity_extraction])  

    domain_examples_key = f"{prompt_domain}_extraction_examples"
    examples = PROMPTS.get(domain_examples_key, PROMPTS[domain_examples_key])
    
    logger.debug(f"Using examples : {domain_examples_key}")
    

    example_context_base = dict(
        tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
        record_delimiter=PROMPTS["DEFAULT_RECORD_DELIMITER"],
        completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
        entity_types=",".join(entity_types),
        language=language,
    )
    
    # add example's format
    examples = examples[0].format(**example_context_base)

    #logger.info(f"Using type of str examples : {type(examples)}")   

    context_base = dict(
        tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
        record_delimiter=PROMPTS["DEFAULT_RECORD_DELIMITER"],
        completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
        entity_types=",".join(entity_types),
        examples=examples,
        language=language,
    )



    continue_prompt = PROMPTS["entiti_continue_extraction"].format(**context_base)
    if_loop_prompt = PROMPTS["entiti_if_loop_extraction"].format(**context_base)

    already_processed = 0
    already_entities = 0
    already_relations = 0

    async def _process_single_content(chunk_key_dp: tuple[str, TextChunkSchema]):
        nonlocal already_processed, already_entities, already_relations
        chunk_key = chunk_key_dp[0]
        chunk_dp = chunk_key_dp[1]
        
        logger.debug(f"DEBUG: Processing chunk: {chunk_key}")
        logger.debug(f"DEBUG: Chunk content: {chunk_dp.get('content', 'NO CONTENT')}")

        content = chunk_dp["content"]
        
        logger.debug(f"Processing content for chunk_key: {chunk_key}")
        logger.debug(f"Content to process: {content[:500]}...")  # Log first 500 chars of content
        
        # Log the context base parameters
        logger.debug(f"Context Base Parameters:")
        logger.debug(f"  Tuple Delimiter: {context_base['tuple_delimiter']}")
        logger.debug(f"  Record Delimiter: {context_base['record_delimiter']}")
        logger.debug(f"  Completion Delimiter: {context_base['completion_delimiter']}")
        logger.debug(f"  Entity Types: {context_base['entity_types']}")
        logger.debug(f"  Language: {context_base['language']}")
        #logger.info(f"  entity_extract_prompt: {entity_extract_prompt}")
        
        hint_prompt = entity_extract_prompt.format(
            **context_base, input_text="{input_text}"
        ).format(**context_base, input_text=content)
        
        #logger.info(f"Formatted Extraction Prompt (first 1000 chars): {hint_prompt[:1000]}...")
        #logger.info(f"Formatted Extraction Prompt (full text): {hint_prompt}...") 
        
        final_result = await use_llm_func(hint_prompt)
        logger.debug(f"Initial LLM Response (first 1000 chars): {final_result[:1000]}...")
        
        history = pack_user_ass_to_openai_messages(hint_prompt, final_result)
        for now_glean_index in range(entity_extract_max_gleaning):
            glean_result = await use_llm_func(continue_prompt, history_messages=history)
            logger.debug(f"Gleaning iteration {now_glean_index + 1} result (first 500 chars): {glean_result[:500]}...")

            history += pack_user_ass_to_openai_messages(continue_prompt, glean_result)
            final_result += glean_result
            
            if now_glean_index == entity_extract_max_gleaning - 1:
                break

            if_loop_result: str = await use_llm_func(
                if_loop_prompt, history_messages=history
            )
            if_loop_result = if_loop_result.strip().strip('"').strip("'").lower()
            logger.debug(f"Should continue gleaning? Answer: {if_loop_result}")
            if if_loop_result != "yes":
                break

        logger.debug(f"Final Complete Result (first 1000 chars): {final_result[:1000]}...")
        
        records = split_string_by_multi_markers(
            final_result,
            [context_base["record_delimiter"], context_base["completion_delimiter"]],
        )
        logger.debug(f"Split Records Count: {len(records)}")
        logger.debug(f"Split Records: {records}")

        maybe_nodes = defaultdict(list)
        maybe_edges = defaultdict(list)

        
        # Collecter tous les noms d'entités par type
        entity_types = {
            "activity": [],
            "user": [],
            "user_preference": [],
            "user_attribute": [],
            "event": [],
            "memo": [],
            "other": []
        }


        for record in records:
            logger.debug(f"Processing Record: {record}")
            record = re.search(r"\((.*)\)", record)
            if record is None:
                logger.warning(f"No parentheses found in record")
                continue
            record = record.group(1)
            record_attributes = split_string_by_multi_markers(
                record, [context_base["tuple_delimiter"]]
            )
            # Strip quotes from each attribute
            record_attributes = [attr.strip().strip('"').strip("'") for attr in record_attributes]
            logger.debug(f"Record Attributes: {record_attributes}")
            
            if_entities = await _handle_single_entity_extraction(
                record_attributes, chunk_key
            )
            if if_entities is not None:
                logger.info(f"Found Entity: {if_entities}")
                
                # Récupérer les metadata spécifiques au type d'entité
                entity_metadata = {}
                if metadata:
                    if prompt_domain == "activity" and "cid" in metadata:
                        entity_metadata["custom_id"] = metadata["cid"]
                    elif prompt_domain == "user" and "user_id" in metadata:
                        entity_metadata["custom_id"] = metadata["user_id"]
                    elif prompt_domain == "event" and "event_id" in metadata:
                        entity_metadata["custom_id"] = metadata["event_id"]
                    elif prompt_domain == "memo" and "custom_id" in metadata:
                        entity_metadata["custom_id"] = metadata["custom_id"]
                
                # Ajouter les metadata à l'entité si disponibles
                if entity_metadata and if_entities["entity_type"] == prompt_domain:
                    if_entities.update(entity_metadata)
                    logger.debug(f"Added metadata to entity: {entity_metadata}")
                
                maybe_nodes[if_entities["entity_name"]].append(if_entities)
                
                # Collecter les entités par type
                if if_entities["entity_type"] == "activity":
                    entity_types["activity"].append(if_entities["entity_name"])
                elif if_entities["entity_type"] == "user":
                    entity_types["user"].append(if_entities["entity_name"])
                elif if_entities["entity_type"] == "user_preference":
                    entity_types["user_preference"].append(if_entities["entity_name"])
                elif if_entities["entity_type"] == "user_attribute":
                    entity_types["user_attribute"].append(if_entities["entity_name"])
                elif if_entities["entity_type"] == "event":
                    entity_types["event"].append(if_entities["entity_name"])
                elif if_entities["entity_type"] == "memo":
                    entity_types["memo"].append(if_entities["entity_name"])
                else:
                    entity_types["other"].append(if_entities["entity_name"])                
                continue

            if_relation = await _handle_single_relationship_extraction(
                record_attributes, chunk_key
            )
            if if_relation is not None:
                logger.debug(f"Found Relationship: {if_relation}")
                maybe_edges[(if_relation["src_id"], if_relation["tgt_id"])].append(
                    if_relation
                )
        
        # Stratégie intelligente de génération de relations
        # 1. Relations utilisateur-préférence
        for user in entity_types["user"]:
            for preference in entity_types["user_preference"]:
                default_relation = {
                    "src_id": user,
                    "tgt_id": preference,
                    "description": f"Préférence personnelle de {user}",
                    "keywords": "préférence personnelle",
                    "weight": 1,
                    "source_id": chunk_key
                }
                maybe_edges[(user, preference)].append(default_relation)
                logger.debug(f"Generated User-Preference Relation: {default_relation}")

        # 2. Lier les attributs utilisateur aux utilisateurs
        for user in entity_types["user"]:
            for attribute in entity_types["user_attribute"]:
                default_relation = {
                    "src_id": user,
                    "tgt_id": attribute,
                    "description": f"Attribut de {user}",
                    "keywords": "information utilisateur",
                    "weight": 1,
                    "source_id": chunk_key
                }
                maybe_edges[(user, attribute)].append(default_relation)
                logger.debug(f"Generated User-Attribute Relation: {default_relation}")

        # 3. Lier les activités à leur contexte principal (s'il existe)
        if len(entity_types["activity"]) > 0:
            main_activity = entity_types["activity"][0]  # Première activité comme contexte principal
            for other_entity in entity_types["other"]:
                default_relation = {
                    "src_id": main_activity,
                    "tgt_id": other_entity,
                    "description": f"Contexte lié à {main_activity}",
                    "keywords": "contexte",
                    "weight": 1,
                    "source_id": chunk_key
                }
                maybe_edges[(main_activity, other_entity)].append(default_relation)
                logger.debug(f"Generated Activity Context Relation: {default_relation}")

        logger.debug(f"DEBUG: Extracted Entities for Chunk {chunk_key}:")
        for entity_type, entities in maybe_nodes.items():
            logger.debug(f"  Entity Type {entity_type}: {len(entities)} entities")
            for entity in entities:
                logger.debug(f"    - {entity}")
        
        logger.debug(f"DEBUG: Extracted Relationships for Chunk {chunk_key}:")
        for relationship_key, relationships in maybe_edges.items():
            logger.debug(f"  Relationship Key {relationship_key}: {len(relationships)} relationships")
            for relationship in relationships:
                logger.debug(f"    - {relationship}")

        already_processed += 1
        already_entities += len(maybe_nodes)
        already_relations += len(maybe_edges)
        now_ticks = PROMPTS["process_tickers"][
            already_processed % len(PROMPTS["process_tickers"])
        ]
        logger.debug(
            f"{now_ticks} Processed {already_processed} chunks, {already_entities} entities(duplicated), {already_relations} relations(duplicated)\r",
            end="",
            flush=True,
        )
        return dict(maybe_nodes), dict(maybe_edges)





    results = []
    for result in tqdm_async(
        asyncio.as_completed([_process_single_content(c) for c in ordered_chunks]),
        total=len(ordered_chunks),
        desc="Extracting entities from chunks",
        unit="chunk",
    ):
        results.append(await result)



    maybe_nodes = defaultdict(list)
    maybe_edges = defaultdict(list)
    for m_nodes, m_edges in results:
        for k, v in m_nodes.items():
            maybe_nodes[k].extend(v)
        for k, v in m_edges.items():
            maybe_edges[tuple(sorted(k))].extend(v)

    logger.info("Inserting entities into storage...")
    logger.info(f"Total maybe_nodes before processing: {len(maybe_nodes)}")
    logger.info(f"maybe_nodes keys: {list(maybe_nodes.keys())}")
    all_entities_data = []
    for result in tqdm_async(
        asyncio.as_completed(
            [
                _merge_nodes_then_upsert(k, v, knowledge_graph_inst, global_config)
                for k, v in maybe_nodes.items()
            ]
        ),
        total=len(maybe_nodes),
        desc="Inserting entities",
        unit="entity",
    ):
        entity_result = await result
        logger.debug(f"Entity merge result: {entity_result}")
        all_entities_data.append(entity_result)

    logger.info(f"Total entities processed: {len(all_entities_data)}")
    #logger.info(f"All entities data: {all_entities_data}")
    
    # Structurer le log avec des couleurs pour plus de lisibilité
    from colorama import Fore, Style
    
    for entity_data in all_entities_data:
        if entity_data:
            entity_type = entity_data.get('entity_type', 'Unknown')
            entity_name = entity_data.get('entity_name', 'N/A')
            
            # Choisir une couleur en fonction du type d'entité
            color = Fore.WHITE
            if entity_type == 'activity':
                color = Fore.GREEN
            elif entity_type == 'user':
                color = Fore.BLUE
            elif entity_type == 'event':
                color = Fore.YELLOW
            elif entity_type == 'user_preference':
                color = Fore.MAGENTA
            
            logger.info(
                f"{color}📦 Entité traitée: "
                f"{Style.BRIGHT}{entity_type}{Style.RESET_ALL} "
                f"{color}→ {Style.BRIGHT}{entity_name}{Style.RESET_ALL}"
            )

    logger.info("Inserting relationships into storage...")
    all_relationships_data = []
    for result in tqdm_async(
        asyncio.as_completed(
            [
                _merge_edges_then_upsert(
                    k[0], k[1], v, knowledge_graph_inst, global_config
                )
                for k, v in maybe_edges.items()
            ]
        ),
        total=len(maybe_edges),
        desc="Inserting relationships",
        unit="relationship",
    ):
        # Log détaillé sur les relations avant insertion
        logger.info(f"Relations à insérer - Source: {k[0]}, Cible: {k[1]}")
        logger.debug(f"Détails des relations : {v}")
        
        all_relationships_data.append(await result)

    if not len(all_entities_data) and not len(all_relationships_data):
        logger.warning(
            "Didn't extract any entities and relationships, maybe your LLM is not working"
        )
        return None

    if not len(all_entities_data):
        logger.warning("Didn't extract any entities")
    if not len(all_relationships_data):
        logger.warning("Didn't extract any relationships")

    if entity_vdb is not None:
        data_for_vdb = {
            compute_mdhash_id(dp["entity_name"], prefix="ent-"): {
                "content": dp["description"],
                "entity_name": dp["entity_name"],
                "entity_type": dp.get("entity_type", "Unknown")
            }
            for dp in all_entities_data
        }
        
        # Log détaillé avant l'insertion dans Milvus
        logger.debug(" Préparation de l'insertion dans Milvus (Entités)")
        logger.debug(f" Nombre d'entités à insérer : {len(data_for_vdb)}")

        # Créer une nouvelle liste pour stocker les entités
        entities_with_description = []

        for entity_id, entity_data in data_for_vdb.items():
            logger.debug(f" ID Entité : {entity_id}")
            logger.debug(f" Données Entité : {entity_data}")
            
            # Récupérer le nœud existant
            existing_node = await knowledge_graph_inst.get_node(entity_data["entity_name"])
            
            # Préparer les données du nœud
            if existing_node is None:
                node_data = {
                    "entity_id": entity_id,
                    "entity_type": entity_data.get("entity_type", "Unknown")
                }
            else:
                # Copier toutes les données existantes et ajouter entity_id et entity_type
                node_data = dict(existing_node)
                node_data["entity_id"] = entity_id
                node_data["entity_type"] = existing_node.get("entity_type", entity_data.get("entity_type", "Unknown"))
                node_data["description"] = existing_node.get("description", entity_data.get("description", "Unknown"))
            
            # Log pour vérification
            logger.debug(f"Données du nœud apres mise à jour : {node_data}")
            
            # Mettre à jour le nœud Neo4j
            await knowledge_graph_inst.upsert_node(
                entity_data["entity_name"], 
                node_data=node_data
            )
            
            # Mettre à jour les données pour Milvus avec entity_type
            entity_data_for_vdb = entity_data.copy()
            entity_data_for_vdb["entity_type"] = node_data["entity_type"]
            data_for_vdb[entity_id] = entity_data_for_vdb

            # Ajouter à la liste des entités avec descriptions
            entity_info = {
                "entity_id": entity_id,
                "content": node_data.get("description", "Pas de description"),
                "entity_name": entity_data["entity_name"],
                "entity_type": node_data["entity_type"]
            }
            entities_with_description.append(entity_info)

        # Logger la liste complète
        logger.debug(f"Entités avec descriptions : {entities_with_description}")

        await entity_vdb.upsert(data_for_vdb)

    if text_chunks is not None and entities_with_description:
        # Prépare un dictionnaire d'entités pour l'insertion dans MongoDB
        entity_chunks_for_mongodb = {
            entity_info["entity_id"]: {
                "_id": entity_info["entity_id"],
                "content": entity_info["content"],
                "entity_name": entity_info["entity_name"],
                "entity_type": entity_info["entity_type"],
                "source": "entity_extraction"
            }
            for entity_info in entities_with_description
        }
        
        # Log détaillé avant l'insertion dans MongoDB
        logger.debug(" Préparation de l'insertion des entités dans MongoDB")
        logger.debug(f" Nombre d'entités à insérer : {len(entity_chunks_for_mongodb)}")

        # Afficher les détails de chaque entité
        # for entity_id, entity_data in entity_chunks_for_mongodb.items():
        #     logger.info(f"🔍 Détails de l'entité : {entity_id}")
        #     logger.info(f"   📋 Nom : {entity_data.get('entity_name', 'N/A')}")
        #     logger.info(f"   🏷️  Type : {entity_data.get('entity_type', 'N/A')}")
        #     logger.info(f"   📝 Contenu : {entity_data.get('content', 'N/A')[:100]}{'...' if len(entity_data.get('content', '')) > 100 else ''}")
        #     logger.info(f"   📦 Source : {entity_data.get('source', 'N/A')}")
        #     logger.info("   " + "-"*50)  # Séparateur visuel
        
        await text_chunks.upsert(entity_chunks_for_mongodb)

    if relationships_vdb is not None:
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
        
        # Mise à jour de Neo4j avec relation_id
        for relation_id, relation_data in data_for_vdb.items():
            logger.debug(f" ID Relation : {relation_id}")
            logger.debug(f" Données Relation : {relation_data}")
            
            # Récupérer l'arête existante
            existing_edge = await knowledge_graph_inst.get_edge(
                relation_data["src_id"], 
                relation_data["tgt_id"]
            )
            
            # Préparer les données de la relation
            if existing_edge is None:
                edge_data = {"relation_id": relation_id}
            else:
                # Copier toutes les données existantes et ajouter relation_id
                edge_data = dict(existing_edge)
                edge_data["relation_id"] = relation_id
            
            # Log pour vérification
            logger.debug(f"Données de la relation avant mise à jour : {edge_data}")
            
            # Mettre à jour la relation Neo4j
            await knowledge_graph_inst.upsert_edge(
                relation_data["src_id"], 
                relation_data["tgt_id"], 
                edge_data=edge_data
            )
        
        await relationships_vdb.upsert(data_for_vdb)
    
    return knowledge_graph_inst


async def kg_query(
    query,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    relationships_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
    global_config: dict,
    hashing_kv: BaseKVStorage = None,
    vdb_filter: Optional[Dict[str, Any]] = None,
) -> str:
    # Handle cache
    use_model_func = global_config["llm_model_func"]
    args_hash = compute_args_hash(query_param.mode, query)
    cached_response, quantized, min_val, max_val = await handle_cache(
        hashing_kv, args_hash, query, query_param.mode
    )
    if cached_response is not None:
        return cached_response

    example_number = global_config["addon_params"].get("example_number", None)
    if example_number and example_number < len(PROMPTS["keywords_extraction_examples"]):
        examples = "\n".join(
            PROMPTS["keywords_extraction_examples"][: int(example_number)]
        )
    else:
        examples = "\n".join(PROMPTS["keywords_extraction_examples"])
    language = global_config["addon_params"].get(
        "language", PROMPTS["DEFAULT_LANGUAGE"]
    )

    # Set mode
    if query_param.mode not in ["local", "global", "hybrid"]:
        logger.error(f"Unknown mode {query_param.mode} in kg_query")
        return PROMPTS["fail_response"]

    # LLM generate keywords
    kw_prompt_temp = PROMPTS["keywords_extraction"]
    kw_prompt = kw_prompt_temp.format(query=query, examples=examples, language=language)
    result = await use_model_func(kw_prompt, keyword_extraction=True)
    logger.debug("kw_prompt result:")
    print(result)
    try:
        # json_text = locate_json_string_body_from_string(result) # handled in use_model_func
        match = re.search(r"\{.*\}", result, re.DOTALL)
        if match:
            result = match.group(0)
            keywords_data = json.loads(result)

            hl_keywords = keywords_data.get("high_level_keywords", [])
            ll_keywords = keywords_data.get("low_level_keywords", [])
        else:
            logger.error("No JSON-like structure found in the result.")
            return PROMPTS["fail_response"]

    # Handle parsing error
    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {e} {result}")
        return PROMPTS["fail_response"]

    # Handdle keywords missing
    if hl_keywords == [] and ll_keywords == []:
        logger.warning("low_level_keywords and high_level_keywords is empty")
        return PROMPTS["fail_response"]
    if ll_keywords == [] and query_param.mode in ["local", "hybrid"]:
        logger.warning("low_level_keywords is empty")
        return PROMPTS["fail_response"]
    else:
        ll_keywords = ", ".join(ll_keywords)
    if hl_keywords == [] and query_param.mode in ["global", "hybrid"]:
        logger.warning("high_level_keywords is empty")
        return PROMPTS["fail_response"]
    else:
        hl_keywords = ", ".join(hl_keywords)

    # Build context
    keywords = [ll_keywords, hl_keywords]
    context = await _build_query_context(
        keywords,
        knowledge_graph_inst,
        entities_vdb,
        relationships_vdb,
        text_chunks_db,
        query_param,
        vdb_filter,
    )

    if query_param.only_need_context:
        return context
    if context is None:
        return PROMPTS["fail_response"]
    sys_prompt_temp = PROMPTS["rag_response"]
    sys_prompt = sys_prompt_temp.format(
        context_data=context, response_type=query_param.response_type
    )
    if query_param.only_need_prompt:
        return sys_prompt
    response = await use_model_func(
        query,
        system_prompt=sys_prompt,
        stream=query_param.stream,
    )
    if isinstance(response, str) and len(response) > len(sys_prompt):
        response = (
            response.replace(sys_prompt, "")
            .replace("user", "")
            .replace("model", "")
            .replace(query, "")
            .replace("<system>", "")
            .replace("</system>", "")
            .strip()
        )

    # Save to cache
    await save_to_cache(
        hashing_kv,
        CacheData(
            args_hash=args_hash,
            content=response,
            prompt=query,
            quantized=quantized,
            min_val=min_val,
            max_val=max_val,
            mode=query_param.mode,
        ),
    )
    return response


async def _build_query_context(
    query: list,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    relationships_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
    vdb_filter: Optional[Dict[str, Any]] = None,
):
    ll_kewwords, hl_keywrds = query[0], query[1]
    if query_param.mode in ["local", "hybrid"]:
        if ll_kewwords == "":
            ll_entities_context, ll_relations_context, ll_text_units_context = (
                "",
                "",
                "",
            )
            warnings.warn(
                "Low Level context is None. Return empty Low entity/relationship/source"
            )
            query_param.mode = "global"
        else:
            (
                ll_entities_context,
                ll_relations_context,
                ll_text_units_context,
            ) = await _get_node_data(
                ll_kewwords,
                knowledge_graph_inst,
                entities_vdb,
                text_chunks_db,
                query_param,
                vdb_filter,
            )
    if query_param.mode in ["global", "hybrid"]:
        if hl_keywrds == "":
            hl_entities_context, hl_relations_context, hl_text_units_context = (
                "",
                "",
                "",
            )
            warnings.warn(
                "High Level context is None. Return empty High entity/relationship/source"
            )
            query_param.mode = "local"
        else:
            (
                hl_entities_context,
                hl_relations_context,
                hl_text_units_context,
            ) = await _get_edge_data(
                hl_keywrds,
                knowledge_graph_inst,
                relationships_vdb,
                text_chunks_db,
                query_param,
                vdb_filter,    
            )
            if (
                hl_entities_context == ""
                and hl_relations_context == ""
                and hl_text_units_context == ""
            ):
                logger.warn("No high level context found. Switching to local mode.")
                query_param.mode = "local"
    if query_param.mode == "hybrid":
        entities_context, relations_context, text_units_context = combine_contexts(
            [hl_entities_context, ll_entities_context],
            [hl_relations_context, ll_relations_context],
            [hl_text_units_context, ll_text_units_context],
        )
    elif query_param.mode == "local":
        entities_context, relations_context, text_units_context = (
            ll_entities_context,
            ll_relations_context,
            ll_text_units_context,
        )
    elif query_param.mode == "global":
        entities_context, relations_context, text_units_context = (
            hl_entities_context,
            hl_relations_context,
            hl_text_units_context,
        )
    return f"""
-----Entities-----
```csv
{entities_context}
```
-----Relationships-----
```csv
{relations_context}
```
-----Sources-----
```csv
{text_units_context}
```
"""


async def _get_node_data(
    query,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
    vdb_filter: Optional[Dict[str, Any]] = None,
):
    
    # get similar entities
    filtered_node_ids = await knowledge_graph_inst.get_filtered_ids(vdb_filter)
    
    # Extraction des node_ids
    filtered_node_ids = filtered_node_ids.get('node_ids', [])

    # Logs pour visualiser filtered_node_ids
    #logger.info(f"Type de filtered_node_ids : {type(filtered_node_ids)}")
    #logger.info(f"Contenu de filtered_node_ids : {filtered_node_ids}")


    if vdb_filter is not None:
        results = await entities_vdb.query(query, top_k=query_param.top_k, vdb_filter=filtered_node_ids)
    else:
        results = await entities_vdb.query(query, top_k=query_param.top_k)
   

    # # Log structuré pour analyser results
    # logger.info("Analyse détaillée des résultats de recherche :")
    # logger.info(f"Nombre total de résultats : {len(results)}")

    # # Afficher les clés présentes dans les résultats
    # if results:
    #     logger.info("Clés disponibles dans les résultats :")
    #     for i, result in enumerate(results, 1):
    #         logger.info(f"Résultat {i} :")
    #         for key, value in result.items():
    #             logger.info(f"  - {key}: {value}")
    #         logger.info("  " + "-"*40)  # Séparateur visuel

    if not len(results):
        return None
    # get entity information
    node_datas = await asyncio.gather(
        *[knowledge_graph_inst.get_node(r["entity_name"]) for r in results]
    )

    if not all([n is not None for n in node_datas]):
        logger.warning("Some nodes are missing, maybe the storage is damaged")

    # get entity degree
    node_degrees = await asyncio.gather(
        *[knowledge_graph_inst.node_degree(r["entity_name"]) for r in results]
    )
    node_datas = [
        {**n, "entity_name": k["entity_name"], "rank": d}
        for k, n, d in zip(results, node_datas, node_degrees)
        if n is not None
    ]  # what is this text_chunks_db doing.  dont remember it in airvx.  check the diagram.
    

    # get entitytext chunk
    use_text_units = await _find_most_related_text_unit_from_entities(
        node_datas, query_param, text_chunks_db, knowledge_graph_inst
    )
    # get relate edges
    use_relations = await _find_most_related_edges_from_entities(
        node_datas, query_param, knowledge_graph_inst
    )

    # Logs pour tracer l'origine de l'erreur
    logger.info(f"Nombre de node_datas : {len(node_datas)}")
    for i, node in enumerate(node_datas):
        logger.info(f"Node {i} - Clés disponibles : {list(node.keys())}")

    # Log avant la construction des relations
    logger.info(f"Nombre de relations : {len(use_relations)}")
    for i, relation in enumerate(use_relations):
        logger.info(f"Relation {i} - Clés disponibles : {list(relation.keys())}")
        # Vérifier spécifiquement l'accès à 'weight'
        try:
            weight = relation['weight']
            logger.info(f"Relation {i} - weight: {weight}")
        except KeyError:
            logger.warning(f"Relation {i} - 'weight' key is missing")

    # build prompt
    entites_section_list = [["id", "entity", "type", "description", "rank"]]
    for i, n in enumerate(node_datas):
        entites_section_list.append(
            [
                i,
                n["entity_name"],
                n.get("entity_type", "UNKNOWN"),
                n.get("description", "UNKNOWN"),
                n["rank"],
            ]
        )
    entities_context = list_of_list_to_csv(entites_section_list)

    relations_section_list = [
        ["id", "source", "target", "description", "keywords", "weight", "rank"]
    ]
    for i, e in enumerate(use_relations):
        relations_section_list.append(
            [
                i,
                e["src_tgt"][0],
                e["src_tgt"][1],
                e["description"],
                e["keywords"],
                e["weight"],
                e["rank"],
            ]
        )
    relations_context = list_of_list_to_csv(relations_section_list)

    text_units_section_list = [["id", "content"]]
    for i, t in enumerate(use_text_units):
        text_units_section_list.append([i, t["content"]])
    text_units_context = list_of_list_to_csv(text_units_section_list)
    return entities_context, relations_context, text_units_context


async def _find_most_related_text_unit_from_entities(
    node_datas: list[dict],
    query_param: QueryParam,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    knowledge_graph_inst: BaseGraphStorage,
):
    text_units = [
        split_string_by_multi_markers(dp["source_id"], [GRAPH_FIELD_SEP])
        for dp in node_datas
    ]
    edges = await asyncio.gather(
        *[knowledge_graph_inst.get_node_edges(dp["entity_name"]) for dp in node_datas]
    )
    all_one_hop_nodes = set()
    for this_edges in edges:
        if not this_edges:
            continue
        all_one_hop_nodes.update([e[1] for e in this_edges])

    all_one_hop_nodes = list(all_one_hop_nodes)
    all_one_hop_nodes_data = await asyncio.gather(
        *[knowledge_graph_inst.get_node(e) for e in all_one_hop_nodes]
    )

    # Add null check for node data
    all_one_hop_text_units_lookup = {
        k: set(split_string_by_multi_markers(v["source_id"], [GRAPH_FIELD_SEP]))
        for k, v in zip(all_one_hop_nodes, all_one_hop_nodes_data)
        if v is not None and "source_id" in v  # Add source_id check
    }

    all_text_units_lookup = {}
    for index, (this_text_units, this_edges) in enumerate(zip(text_units, edges)):
        for c_id in this_text_units:
            if c_id not in all_text_units_lookup:
                all_text_units_lookup[c_id] = {
                    "data": await text_chunks_db.get_by_id(c_id),
                    "order": index,
                    "relation_counts": 0,
                }

            if this_edges:
                for e in this_edges:
                    if (
                        e[1] in all_one_hop_text_units_lookup
                        and c_id in all_one_hop_text_units_lookup[e[1]]
                    ):
                        all_text_units_lookup[c_id]["relation_counts"] += 1

    # Filter out None values and ensure data has content
    all_text_units = [
        {"id": k, **v}
        for k, v in all_text_units_lookup.items()
        if v is not None and v.get("data") is not None and "content" in v["data"]
    ]

    if not all_text_units:
        logger.warning("No valid text units found")
        return []

    all_text_units = sorted(
        all_text_units, key=lambda x: (x["order"], -x["relation_counts"])
    )

    all_text_units = truncate_list_by_token_size(
        all_text_units,
        key=lambda x: x["data"]["content"],
        max_token_size=query_param.max_token_for_text_unit,
    )

    all_text_units = [t["data"] for t in all_text_units]
    return all_text_units


async def _find_most_related_edges_from_entities(
    node_datas: list[dict],
    query_param: QueryParam,
    knowledge_graph_inst: BaseGraphStorage,
):
    all_related_edges = await asyncio.gather(
        *[knowledge_graph_inst.get_node_edges(dp["entity_name"]) for dp in node_datas]
    )
    all_edges = []
    seen = set()

    for this_edges in all_related_edges:
        for e in this_edges:
            sorted_edge = tuple(sorted(e))
            if sorted_edge not in seen:
                seen.add(sorted_edge)
                all_edges.append(sorted_edge)

    all_edges_pack = await asyncio.gather(
        *[knowledge_graph_inst.get_edge(e[0], e[1]) for e in all_edges]
    )
    all_edges_degree = await asyncio.gather(
        *[knowledge_graph_inst.edge_degree(e[0], e[1]) for e in all_edges]
    )
    all_edges_data = [
        {"src_tgt": k, "rank": d, **v}
        for k, v, d in zip(all_edges, all_edges_pack, all_edges_degree)
        if v is not None
    ]
    all_edges_data = sorted(
        all_edges_data, key=lambda x: (x["rank"], x["weight"]), reverse=True
    )
    all_edges_data = truncate_list_by_token_size(
        all_edges_data,
        key=lambda x: x["description"],
        max_token_size=query_param.max_token_for_global_context,
    )
    return all_edges_data


async def _get_edge_data(
    keywords,
    knowledge_graph_inst: BaseGraphStorage,
    relationships_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
    vdb_filter: Optional[Dict[str, Any]] = None,
):



    # get similar entities
    filtered_edge_ids = await knowledge_graph_inst.get_filtered_ids(vdb_filter)
    
    # Extraction des node_ids
    filtered_edge_ids = filtered_edge_ids.get('relation_ids', [])

    # Logs pour visualiser filtered_node_ids
    #logger.info(f"Type de filtered_node_ids : {type(filtered_node_ids)}")
    #logger.info(f"Contenu de filtered_edge_ids : {filtered_edge_ids}")


    if vdb_filter is not None:
        results = await relationships_vdb.query(keywords, top_k=query_param.top_k, vdb_filter=filtered_edge_ids)
    else:
        results = await relationships_vdb.query(keywords, top_k=query_param.top_k)
    
    if not len(results):
        return "", "", ""

    edge_datas = await asyncio.gather(
        *[knowledge_graph_inst.get_edge(r["src_id"], r["tgt_id"]) for r in results]
    )

    if not all([n is not None for n in edge_datas]):
        logger.warning("Some edges are missing, maybe the storage is damaged")
    edge_degree = await asyncio.gather(
        *[knowledge_graph_inst.edge_degree(r["src_id"], r["tgt_id"]) for r in results]
    )
    edge_datas = [
        {"src_id": k["src_id"], "tgt_id": k["tgt_id"], "rank": d, **v}
        for k, v, d in zip(results, edge_datas, edge_degree)
        if v is not None
    ]
    edge_datas = sorted(
        edge_datas, key=lambda x: (x["rank"], x["weight"]), reverse=True
    )
    edge_datas = truncate_list_by_token_size(
        edge_datas,
        key=lambda x: x["description"],
        max_token_size=query_param.max_token_for_global_context,
    )

    use_entities = await _find_most_related_entities_from_relationships(
        edge_datas, query_param, knowledge_graph_inst
    )
    use_text_units = await _find_related_text_unit_from_relationships(
        edge_datas, query_param, text_chunks_db, knowledge_graph_inst
    )
    logger.debug(
        f"Global query uses {len(use_entities)} entites, {len(edge_datas)} relations, {len(use_text_units)} text units"
    )

    relations_section_list = [
        ["id", "source", "target", "description", "keywords", "weight", "rank"]
    ]
    for i, e in enumerate(edge_datas):
        relations_section_list.append(
            [
                i,
                e["src_id"],
                e["tgt_id"],
                e["description"],
                e["keywords"],
                e["weight"],
                e["rank"],
            ]
        )
    relations_context = list_of_list_to_csv(relations_section_list)

    entites_section_list = [["id", "entity", "type", "description", "rank"]]
    for i, n in enumerate(use_entities):
        entites_section_list.append(
            [
                i,
                n["entity_name"],
                n.get("entity_type", "UNKNOWN"),
                n.get("description", "UNKNOWN"),
                n["rank"],
            ]
        )
    entities_context = list_of_list_to_csv(entites_section_list)

    text_units_section_list = [["id", "content"]]
    for i, t in enumerate(use_text_units):
        text_units_section_list.append([i, t["content"]])
    text_units_context = list_of_list_to_csv(text_units_section_list)
    return entities_context, relations_context, text_units_context


async def _find_most_related_entities_from_relationships(
    edge_datas: list[dict],
    query_param: QueryParam,
    knowledge_graph_inst: BaseGraphStorage,
):
    entity_names = []
    seen = set()

    for e in edge_datas:
        if e["src_id"] not in seen:
            entity_names.append(e["src_id"])
            seen.add(e["src_id"])
        if e["tgt_id"] not in seen:
            entity_names.append(e["tgt_id"])
            seen.add(e["tgt_id"])

    node_datas = await asyncio.gather(
        *[knowledge_graph_inst.get_node(entity_name) for entity_name in entity_names]
    )

    node_degrees = await asyncio.gather(
        *[knowledge_graph_inst.node_degree(entity_name) for entity_name in entity_names]
    )
    node_datas = [
        {**n, "entity_name": k, "rank": d}
        for k, n, d in zip(entity_names, node_datas, node_degrees)
    ]

    node_datas = truncate_list_by_token_size(
        node_datas,
        key=lambda x: x["description"],
        max_token_size=query_param.max_token_for_local_context,
    )

    return node_datas


async def _find_related_text_unit_from_relationships(
    edge_datas: list[dict],
    query_param: QueryParam,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    knowledge_graph_inst: BaseGraphStorage,
):
    text_units = [
        split_string_by_multi_markers(dp["source_id"], [GRAPH_FIELD_SEP])
        for dp in edge_datas
    ]
    all_text_units_lookup = {}

    for index, unit_list in enumerate(text_units):
        for c_id in unit_list:
            if c_id not in all_text_units_lookup:
                chunk_data = await text_chunks_db.get_by_id(c_id)
                # Only store valid data
                if chunk_data is not None and "content" in chunk_data:
                    all_text_units_lookup[c_id] = {
                        "data": chunk_data,
                        "order": index,
                    }

    if not all_text_units_lookup:
        logger.warning("No valid text chunks found")
        return []

    all_text_units = [{"id": k, **v} for k, v in all_text_units_lookup.items()]
    all_text_units = sorted(all_text_units, key=lambda x: x["order"])

    # Ensure all text chunks have content
    valid_text_units = [
        t for t in all_text_units if t["data"] is not None and "content" in t["data"]
    ]

    if not valid_text_units:
        logger.warning("No valid text chunks after filtering")
        return []

    truncated_text_units = truncate_list_by_token_size(
        valid_text_units,
        key=lambda x: x["data"]["content"],
        max_token_size=query_param.max_token_for_text_unit,
    )

    all_text_units: list[TextChunkSchema] = [t["data"] for t in truncated_text_units]

    return all_text_units


def combine_contexts(entities, relationships, sources):
    # Function to extract entities, relationships, and sources from context strings
    hl_entities, ll_entities = entities[0], entities[1]
    hl_relationships, ll_relationships = relationships[0], relationships[1]
    hl_sources, ll_sources = sources[0], sources[1]
    # Combine and deduplicate the entities
    combined_entities = process_combine_contexts(hl_entities, ll_entities)

    # Combine and deduplicate the relationships
    combined_relationships = process_combine_contexts(
        hl_relationships, ll_relationships
    )

    # Combine and deduplicate the sources
    combined_sources = process_combine_contexts(hl_sources, ll_sources)

    return combined_entities, combined_relationships, combined_sources


async def naive_query(
    query,
    chunks_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
    global_config: dict,
    hashing_kv: BaseKVStorage = None,
):
    # Handle cache
    use_model_func = global_config["llm_model_func"]
    args_hash = compute_args_hash(query_param.mode, query)
    cached_response, quantized, min_val, max_val = await handle_cache(
        hashing_kv, args_hash, query, query_param.mode
    )
    if cached_response is not None:
        return cached_response

    results = await chunks_vdb.query(query, top_k=query_param.top_k)
    if not len(results):
        return PROMPTS["fail_response"]

    chunks_ids = [r["id"] for r in results]
    chunks = await text_chunks_db.get_by_ids(chunks_ids)

    # Filter out invalid chunks
    valid_chunks = [
        chunk for chunk in chunks if chunk is not None and "content" in chunk
    ]

    if not valid_chunks:
        logger.warning("No valid chunks found after filtering")
        return PROMPTS["fail_response"]

    maybe_trun_chunks = truncate_list_by_token_size(
        valid_chunks,
        key=lambda x: x["content"],
        max_token_size=query_param.max_token_for_text_unit,
    )

    if not maybe_trun_chunks:
        logger.warning("No chunks left after truncation")
        return PROMPTS["fail_response"]

    logger.debug(f"Truncate {len(chunks)} to {len(maybe_trun_chunks)} chunks")
    section = "\n--New Chunk--\n".join([c["content"] for c in maybe_trun_chunks])

    if query_param.only_need_context:
        return section

    sys_prompt_temp = PROMPTS["naive_rag_response"]
    sys_prompt = sys_prompt_temp.format(
        content_data=section, response_type=query_param.response_type
    )

    if query_param.only_need_prompt:
        return sys_prompt

    response = await use_model_func(
        query,
        system_prompt=sys_prompt,
    )

    if len(response) > len(sys_prompt):
        response = (
            response[len(sys_prompt) :]
            .replace(sys_prompt, "")
            .replace("user", "")
            .replace("model", "")
            .replace(query, "")
            .replace("<system>", "")
            .replace("</system>", "")
            .strip()
        )

    # Save to cache
    await save_to_cache(
        hashing_kv,
        CacheData(
            args_hash=args_hash,
            content=response,
            prompt=query,
            quantized=quantized,
            min_val=min_val,
            max_val=max_val,
            mode=query_param.mode,
        ),
    )

    return response
