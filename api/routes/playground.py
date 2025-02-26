from os import getenv
from phi.playground import Playground

from agents.example import get_example_agent
from agents.web import get_web_searcher
from agents.lightrag_reader import get_lightrag_reader

######################################################
## Router for the agent playground
######################################################

example_agent = get_example_agent(debug_mode=True)
web_agent = get_web_searcher(debug_mode=True)
lightning_agent = get_lightrag_reader(debug_mode=True)

# Create a playground instance
playground = Playground(agents=[example_agent, web_agent, lightning_agent])

# Log the playground endpoint with phidata.app
if getenv("RUNTIME_ENV") == "dev":
    playground.create_endpoint("http://localhost:8000")

playground_router = playground.get_router()
