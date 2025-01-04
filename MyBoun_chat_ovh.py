"""
Script Python pour envoyer un message à un chatbot via une API REST.

### Règles d'utilisation :
1. **Prérequis** :
   - Python installé (3.6 ou supérieur recommandé).
   - Les bibliothèques `requests` et `argparse` doivent être disponibles.
     Vous pouvez installer `requests` avec la commande : `pip install requests`.

2. **Exécution** :
   - Ce script est conçu pour être exécuté via la ligne de commande.
   - Fournissez le message à envoyer au chatbot comme argument principal.
     Exemple : `python script.py "mon nom est Vinh"`

3. **Options supplémentaires** :
   - `--agent_id` : ID de l'agent utilisé par le chatbot (valeur par défaut : "example-agent").
   - `--session_id` : ID de session pour suivre une conversation continue (valeur par défaut : "vinh_session_id").
   - `--user_id` : Identifiant unique de l'utilisateur (valeur par défaut : "vinh_id").
   - `--url` : URL de l'API REST du chatbot (valeur par défaut : "http://vps-af24e24d.vps.ovh.net/v1/playground/agent/run").

4. **Sortie** :
   - Le script affiche la réponse du chatbot ou un message d'erreur en cas de problème.

5. **Exemples d'exécution** :
   - Simple : `python MyBoun_chat.py "mon nom est Vinh"`
   - Personnalisé : `python MyBoun_chat.py "Bonjour" --agent_id "custom-agent" --session_id "session123" --url "http://mon-api"`

6. **Cas d'erreur** :
   - Si l'API retourne une erreur HTTP, le script affichera un message décrivant l'erreur.

### Auteur : Vous
"""

import argparse
import requests
import json

def envoyer_message(message, agent_id="example", session_id="vinh_session_id_2", user_id="vinh_id-2", url="http://vps-af24e24d.vps.ovh.net/v1/playground/agent/run"):
    """
    Envoie un message à l'API de chat et récupère la réponse.
    
    :param message: Le message à envoyer
    :param agent_id: ID de l'agent
    :param session_id: ID de session
    :param user_id: ID de l'utilisateur
    :param url: URL de l'API de chat
    :return: La réponse complète du chat
    """
    # Corps de la requête
    payload = {
        "message": message,
        "agent_id": agent_id,
        "stream": False,
        "monitor": False,
        "session_id": session_id,
        "user_id": user_id
    }

    # En-têtes HTTP
    headers = {
        "accept": "application/json",
        "Content-Type": "application/json"
    }

    try:
        # Envoi de la requête POST
        response = requests.post(url, headers=headers, json=payload)
        
        # Vérification du succès de la requête
        response.raise_for_status()
        
        # Essayer de décoder la réponse
        try:
            # La réponse est une chaîne JSON, donc on doit la parser deux fois
            response_text = response.text
            data = json.loads(response_text)
            
            # Si c'est un dictionnaire JSON
            if isinstance(data, dict):
                # Vérifier différentes structures possibles
                if 'content' in data:
                    return data['content']
                
                if 'messages' in data:
                    for message in data['messages']:
                        if isinstance(message, dict) and message.get('role') == 'assistant':
                            return message.get('content', '')
            
            # Si c'est une chaîne JSON, la parser à nouveau
            if isinstance(data, str):
                nested_data = json.loads(data)
                if isinstance(nested_data, dict):
                    if 'content' in nested_data:
                        return nested_data['content']
                    
                    if 'messages' in nested_data:
                        for message in nested_data['messages']:
                            if isinstance(message, dict) and message.get('role') == 'assistant':
                                return message.get('content', '')
            
            # Si aucun contenu n'est trouvé, retourner le texte brut
            return response_text
        
        except json.JSONDecodeError:
            # Si la réponse n'est pas du JSON, retourner le texte brut
            return response.text
    
    except requests.exceptions.RequestException as e:
        return f"Erreur lors de l'envoi du message : {e}"

def main():
    # Configurer le parser d'arguments
    parser = argparse.ArgumentParser(description="Envoyer un message à l'API de chat")
    parser.add_argument("message", type=str, help="Le message à envoyer")
    parser.add_argument("--agent_id", type=str, default="example-agent", help="ID de l'agent")
    parser.add_argument("--session_id", type=str, default="vinh_session_id", help="ID de session")
    parser.add_argument("--user_id", type=str, default="vinh_id", help="ID de l'utilisateur")
    #parser.add_argument("--url", type=str, default="http://vps-af24e24d.vps.ovh.net/v1/playground/agent/run", help="URL de l'API de chat")
    parser.add_argument("--url", type=str, default="http://localhost:8000/v1/playground/agent/run", help="URL de l'API de chat")
    
    # Analyser les arguments
    args = parser.parse_args()
    
    # Envoyer le message et afficher la réponse
    reponse = envoyer_message(args.message, args.agent_id, args.session_id, args.user_id, args.url)
    print(reponse)

if __name__ == "__main__":
    main()