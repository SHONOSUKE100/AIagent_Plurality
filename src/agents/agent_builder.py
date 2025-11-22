"""
Agentの作成スクリプト
"""

from oasis import SocialAgent, AgentGraph
from camel.models import ModelFactory
from camel.types import ModelPlatformType, ModelType
from oasis import ActionType, LLMAction, ManualAction
from dataclasses import dataclass, asdict, field
import warnings
from camel.prompts import TextPrompt
import json

@dataclass
class UserInfo:
    user_name: str = "user_name"
    name: str = "user_name"
    occupation: str = "学生"
    age: int = 20
    hobbies: list = field(default_factory=list)
    residence: str = "東京"
    description: str = "毎日２キロ走ってます。"
    recsys_type: str = "twitter"
    is_controllable: bool = False

    def to_custom_system_message(self, user_info_template: TextPrompt) -> str:
        required_keys = user_info_template.key_words
        profile = asdict(self)

        info_keys = set(profile.keys())
        missing = required_keys - info_keys
        extra = info_keys - required_keys

        if missing:
            raise ValueError(
                f"Missing required keys in UserInfo.profile: {missing}")
        if extra:
            warnings.warn(f"Extra keys not used in UserInfo.profile: {extra}")

        return user_info_template.format(**profile)

    def to_system_message(self) -> str:
        if self.recsys_type != "reddit":
            return self.to_twitter_system_message()
        else:
            return self.to_reddit_system_message()

    def to_twitter_system_message(self) -> str:
        hobbies_str = ", ".join(self.hobbies)
        description = (
            f"You are a {self.age} year old {self.occupation} living in {self.residence}. "
            f"Your hobbies are {hobbies_str}. "
            f"{self.description}"
        )

        system_content = f"""
# OBJECTIVE
You're a Twitter user, and I'll present you with some posts. After you see the posts, choose some actions from the following functions.

# SELF-DESCRIPTION
Your actions should be consistent with your self-description and personality.
{description}

# RESPONSE METHOD
Please perform actions by tool calling.
        """

        return system_content


def generate_x_agent_graph(
    profile_path,
    model,
    available_actions
):
    with open(profile_path, 'r', encoding='utf-8') as f:
        agent_info = json.load(f)
    agent_graph = AgentGraph()
    for agent_id in range(len(agent_info)):
        profile = {
            "nodes": [],
            "edges": [],
            "other_info": {},
        }
        
        profile["other_info"]["user_profile"] = agent_info[agent_id]

        user_info = UserInfo(
            user_name = agent_info[agent_id]["user_name"],
            name = agent_info[agent_id]["name"],
            occupation = agent_info[agent_id]["occupation"],
            age = agent_info[agent_id]["age"],
            hobbies = agent_info[agent_id]["hobbies"],
            residence = agent_info[agent_id]["residence"],
            description = agent_info[agent_id]["description"],
            recsys_type = "twitter"
        )

        agent = SocialAgent(
            agent_id=agent_id,
            user_info=user_info,
            model=model,
            agent_graph=agent_graph,
            available_actions=available_actions,
        )

        agent_graph.add_agent(agent)
    
    return agent_graph

agent_graph = generate_x_agent_graph(
    profile_path="../data/persona/persona.json",
    model=model,
    available_actions=available_actions
)

