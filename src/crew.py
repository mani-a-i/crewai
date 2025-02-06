from crewai import Agent, Crew
from crewai.project import CrewBase,agent
from models import llm_registry as global_llm


@CrewBase
class IVRBot():
    """ IVRBot Crew"""
    agents_config = "config/agents.yaml"

    @agent
    def Customer_support_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['Customer_support_agent'],
            verbose=True,
            llm=global_llm.gemini_flash
        )