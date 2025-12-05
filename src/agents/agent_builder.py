"""Utilities for constructing OASIS social agents from persona profiles."""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Iterable, List, Mapping, Sequence
import json
import warnings

from camel.models import ModelFactory
from camel.prompts import TextPrompt
from camel.types import ModelPlatformType, ModelType
from oasis import ActionType, AgentGraph, SocialAgent


Persona = Mapping[str, object]


@dataclass
class UserInfo:
    """Container describing an agent persona for OASIS social agents."""

    user_name: str = "user_name"
    name: str = "user_name"
    occupation: str = "学生"
    age: int = 20
    hobbies: List[str] = field(default_factory=list)
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
            raise ValueError(f"Missing required keys in UserInfo.profile: {missing}")
        if extra:
            warnings.warn(f"Extra keys not used in UserInfo.profile: {extra}")

        return user_info_template.format(**profile)

    def to_system_message(self) -> str:
        if self.recsys_type != "reddit":
            return self.to_twitter_system_message()
        return self.to_reddit_system_message()

    def to_twitter_system_message(self) -> str:
        hobbies_str = ", ".join(self.hobbies)
        description = (
            f"You are a {self.age} year old {self.occupation} living in {self.residence}. "
            f"Your hobbies are {hobbies_str}. "
            f"{self.description}"
        )

        system_content = """
# OBJECTIVE
You're a Twitter user, and I'll present you with some posts. After you see the posts, choose some actions from the following functions.

# SELF-DESCRIPTION
Your actions should be consistent with your self-description and personality.
{description}

# RESPONSE METHOD
Please perform actions by tool calling.
        """

        return system_content

    def to_reddit_system_message(self) -> str:
        description = (
            f"You are a {self.age} year old {self.occupation} participating in Reddit communities. "
            f"You live in {self.residence} and enjoy {', '.join(self.hobbies)}. "
            f"{self.description}"
        )

        system_content = """
# OBJECTIVE
You're a Reddit user engaging in subreddit discussions. Please interact with the feed according to the available actions.

# SELF-DESCRIPTION
Your actions should reflect your background and preferences.
{description}

# RESPONSE METHOD
Please perform actions by tool calling.
        """

        return system_content


DEFAULT_AVAILABLE_ACTIONS: Sequence[ActionType] = (
    ActionType.LIKE_POST,
    ActionType.DISLIKE_POST,
    ActionType.CREATE_POST,
    ActionType.CREATE_COMMENT,
    ActionType.LIKE_COMMENT,
    ActionType.DISLIKE_COMMENT,
    ActionType.SEARCH_POSTS,
    ActionType.SEARCH_USER,
    ActionType.TREND,
    ActionType.REFRESH,
    ActionType.DO_NOTHING,
    ActionType.FOLLOW,
    ActionType.MUTE,
)


def load_personas(profile_path: Path | str) -> List[Persona]:
    """Read persona definitions from disk."""

    path = Path(profile_path)
    with path.open("r", encoding="utf-8") as profile_file:
        persona_data = json.load(profile_file)

    if not isinstance(persona_data, list):
        raise ValueError(f"Persona file must contain a list, found {type(persona_data)!r}")

    return persona_data


def create_default_model(
    *,
    temperature: float = 0.0,
    model_type: ModelType = ModelType.GPT_4O,
    model_platform: ModelPlatformType = ModelPlatformType.OPENAI,
    model_config_dict: Mapping[str, object] | None = None,
):
    """Instantiate the default CAMEL model used by the notebook prototype."""

    config = {"temperature": temperature}
    if model_config_dict:
        config.update(model_config_dict)

    return ModelFactory.create(
        model_platform=model_platform,
        model_type=model_type,
        model_config_dict=config,
    )


def _normalize_persona(persona: Persona, fallback_id: int) -> UserInfo:
    """Convert a persona mapping into the UserInfo schema expected by OASIS."""

    user_name = str(persona.get("user_name") or persona.get("name") or f"user_{fallback_id:04d}")
    name = str(persona.get("name") or user_name)
    description = str(
        persona.get("description")
        or persona.get("profile")
        or "This agent is ready to join the simulation."
    )

    hobbies_value = persona.get("hobbies") or []
    if isinstance(hobbies_value, str):
        hobbies = [hobbies_value]
    elif isinstance(hobbies_value, Iterable):
        hobbies = [str(hobby) for hobby in hobbies_value]
    else:
        hobbies = []

    return UserInfo(
        user_name=user_name,
        name=name,
        occupation=str(persona.get("occupation") or "会社員"),
        age=int(persona.get("age") or 20),
        hobbies=hobbies,
        residence=str(persona.get("residence") or persona.get("location") or "東京"),
        description=description,
    )


def build_agent_graph(
    personas: Sequence[Persona],
    model,
    available_actions: Sequence[ActionType] | None = None,
) -> AgentGraph:
    """Create an OASIS :class:`AgentGraph` from persona dictionaries."""

    actions = list(available_actions or DEFAULT_AVAILABLE_ACTIONS)
    agent_graph = AgentGraph()

    for agent_id, persona in enumerate(personas):
        user_info = _normalize_persona(persona, fallback_id=agent_id)
        agent = SocialAgent(
            agent_id=agent_id,
            user_info=user_info,
            model=model,
            agent_graph=agent_graph,
            available_actions=actions,
        )

        agent_graph.add_agent(agent)

    return agent_graph


def build_agent_graph_from_file(
    profile_path: Path | str,
    model,
    available_actions: Sequence[ActionType] | None = None,
) -> AgentGraph:
    """Load personas from ``profile_path`` and build an agent graph."""

    personas = load_personas(profile_path)
    return build_agent_graph(personas, model, available_actions=available_actions)

