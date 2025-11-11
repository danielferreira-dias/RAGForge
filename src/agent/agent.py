
from pydantic_ai import Agent
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.providers.google import GoogleProvider

import os
from dotenv import load_dotenv

from pydantic import Field

load_dotenv()
class AgentOutput:
    agent_answer: str = Field(description='Agent answer')

class AgentRAG():
    provider : GoogleProvider = GoogleProvider(api_key=os.getenv('GOOGLE_API_KEY'))

    def __init__(self, model : str = 'gemini-2.5-pro'):
        self.model = GoogleModel(model, provider=self.provider)
        self.agent = Agent(self.model, output_type = AgentOutput, system_prompt= 'You are an AI Agent assistant;')

    async def invoke_agent(self, user_query : str):
        return await self.agent.run(user_query)