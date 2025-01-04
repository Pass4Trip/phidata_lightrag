from typing import List, Dict, Optional, Any
from lightrag.utils import logger

class ActivityCategoriesManager:
    def __init__(self):
        # Liste des catégories prédéfinies
        self._categories = {
            "Restauration": [
                "restaurant", "café", "bar", "bistro", "brasserie", 
                "gastronomie", "cuisine", "repas", "déjeuner", "dîner"
            ],
            "Culture et Loisirs": [
                "concert", "exposition", "musée", "théâtre", "cinéma", 
                "spectacle", "art", "festival", "événement culturel"
            ],
            "Sport et Fitness": [
                "sport", "gym", "fitness", "match", "compétition", 
                "entraînement", "course", "cyclisme", "natation"
            ],
            "Voyage et Tourisme": [
                "voyage", "tourisme", "excursion", "visite", "randonnée", 
                "séjour", "destination", "circuit"
            ],
            "Formation et Éducation": [
                "cours", "formation", "atelier", "séminaire", "conférence", 
                "workshop", "apprentissage", "école"
            ],
            "Bien-être et Santé": [
                "spa", "massage", "yoga", "méditation", "relaxation", 
                "soins", "bien-être", "santé"
            ],
            "Événements Professionnels": [
                "networking", "conférence", "salon", "réunion", "séminaire", 
                "business", "professionnel", "entreprise"
            ],
            "Unknown": []  # Catégorie explicite pour les activités non classées
        }
        
        # Catégorie par défaut si aucune correspondance n'est trouvée
        self._default_category = "Unknown"
    
    def get_category(self, description: str) -> str:
        """
        Détermine la catégorie d'une activité en fonction de sa description
        """
        description_lower = description.lower()
        
        logger.info(f"🔍 Recherche de catégorie pour la description : {description_lower}")
        
        for category, keywords in self._categories.items():
            # Ignorer la catégorie "Unknown"
            if category == "Unknown":
                continue
            
            matching_keywords = [keyword for keyword in keywords if keyword in description_lower]
            if matching_keywords:
                logger.info(f"✅ Catégorie trouvée : {category}")
                logger.info(f"   Mots-clés correspondants : {matching_keywords}")
                return category
        
        logger.warning(f"❌ Aucune catégorie trouvée, utilisation de la catégorie par défaut : {self._default_category}")
        return self._default_category
    
    def add_category(self, category_name: str, keywords: List[str]):
        """
        Permet d'ajouter une nouvelle catégorie personnalisée
        """
        self._categories[category_name] = keywords
    
    def list_categories(self) -> List[str]:
        """
        Retourne la liste de toutes les catégories
        """
        return list(self._categories.keys())
    
    async def ensure_categories_in_graph(self, graph_storage: Any) -> None:
        """
        Vérifie et crée les nœuds de catégories dans le graphe s'ils n'existent pas
        
        :param graph_storage: Instance de stockage de graphe (Neo4j ou équivalent)
        """
        # Liste des catégories à créer
        categories_to_create = self.list_categories()
        
        async def create_categories_tx(tx):
            for category in categories_to_create:
                if category == "Unknown":
                    continue
                
                # Requête pour créer le nœud de catégorie si non existant
                query = f"""
                MERGE (cat:ActivityCategory {{name: $category_name}})
                RETURN cat.name
                """
                
                try:
                    result = await tx.run(query, category_name=category)
                    await result.consume()
                except Exception as e:
                    logger.warning(f"❌ Erreur lors de la création de la catégorie {category} : {e}")
        
        try:
            async with graph_storage._driver.session() as session:
                await session.execute_write(create_categories_tx)
            logger.info("✅ Toutes les catégories d'activités ont été initialisées dans le graphe.")
        except Exception as e:
            logger.warning(f"❌ Erreur lors de l'initialisation des catégories : {e}")

# Instance globale pour être utilisée facilement
activity_categories_manager = ActivityCategoriesManager()
