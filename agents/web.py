from typing import Optional

from phi.agent import Agent
from phi.model.openai import OpenAIChat


from phi.knowledge.agent import AgentKnowledge
from phi.storage.agent.postgres import PgAgentStorage
from phi.tools.duckduckgo import DuckDuckGo
from phi.vectordb.pgvector import PgVector, SearchType

from agents.settings import agent_settings
from db.session import db_url


web_agent_storage = PgAgentStorage(table_name="web_agent_sessions", db_url=db_url)
web_agent_knowledge = AgentKnowledge(vector_db=PgVector(table_name="web_agent_knowledge", db_url=db_url, search_type=SearchType.hybrid))

 
def get_web_searcher(
    model_id: Optional[str] = None,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    debug_mode: bool = False,
) -> Agent:
    return Agent(
        name="Web Searcher",
        agent_id="web-searcher",
        session_id=session_id,
        user_id=user_id,
        # The model to use for the agent
        model=OpenAIChat(
            id=model_id or agent_settings.gpt_4,
            max_tokens=agent_settings.default_max_completion_tokens,
            temperature=agent_settings.default_temperature,
        ),
        #role="Tu es un expert d'internet et tu donnera les informations pour répondre aux questions autres que les restaurants à Lyon qui sont à la charge de RAG Restaurant Searcher.", 
        role="Tu es un agent d'une Team qui dispose des capacités à rechercher des informations sur le Web. Tu dois renvoyer tes résultats à Agent Leader",  
        instructions=[
            "To answer the user's question, first search the web for information by breaking down the user's question into smaller queries.",
            "Make sure you cover all the aspects of the question.",
            "Important: \n"
            " - Focus on legitimate sources\n"
            " - Always provide sources and the links to the information you used to answer the question\n"
            " - If you cannot find the answer, say so and ask the user to provide more details.",
            "Keep your answers concise and engaging.", 
        ],
        tools=[DuckDuckGo()],
        add_datetime_to_instructions=True,
        markdown=True,
        # Show tool calls in the response
        show_tool_calls=True,
        # Add the current date and time to the instructions
        # Store agent sessions in the database
        storage=web_agent_storage,
        # Enable read the chat history from the database
        read_chat_history=True,
        # Store knowledge in a vector database
        knowledge=web_agent_knowledge,
        # Enable searching the knowledge base
        #search_knowledge=True,
        # Enable monitoring on phidata.app
        monitoring=True,
        # Show debug logs
        debug_mode=True,
    )