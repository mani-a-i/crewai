from crewai import Agent, Crew, Task, Process
from crewai.project import CrewBase, agent, task, crew
from models import llm_registry as global_llm


@CrewBase
class IVRBot():
    """ IVRBot Crew"""
    agents_config = "config/agents.yaml"
    task_config = "config/tasks.yaml"

    @agent
    def Customer_support_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['Customer_support_agent'],
            verbose=True,
            llm=global_llm.gemini_flash
        )
    
    @task
    def Customer_support_agent_task(self) -> Task:
        return Task(
            config=self.task_config['Customer_support_agent_task']
        )
    
    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.Customer_support_agent,
            tasks=self.Customer_support_agent_task,
            process= Process.sequential,
            verbose=True
        )
    
    
    
