# Configuration des Catégories d'Activités

## Présentation

Le fichier `activity_categories.py` permet de gérer dynamiquement les catégories d'activités utilisées lors de l'insertion de données dans LightRAG.

## Personnalisation des Catégories

### Ajouter une Nouvelle Catégorie

```python
from lightrag.config.activity_categories import activity_categories_manager

# Ajouter une nouvelle catégorie
activity_categories_manager.add_category(
    "Jardinage", 
    ["jardin", "plantes", "horticulture", "plantation", "jardinage"]
)
```

### Lister les Catégories Disponibles

```python
# Obtenir la liste des catégories
categories = activity_categories_manager.list_categories()
print(categories)
```

## Fonctionnement Interne

- Chaque catégorie est définie par un ensemble de mots-clés
- La méthode `get_category()` analyse la description de l'activité
- Si aucune correspondance n'est trouvée, la catégorie "Autres" est utilisée

## Personnalisation Avancée

Vous pouvez modifier directement le dictionnaire `_categories` dans `activity_categories.py` pour une personnalisation complète.

## Bonnes Pratiques

- Utilisez des mots-clés génériques et représentatifs
- Évitez les mots-clés trop spécifiques
- Pensez à couvrir différentes variations linguistiques
