"""
Agentの作成スクリプト
"""

from oasis import SocialAgent
from camel.models import ModelFactory
from camel.types import ModelPlatformType, ModelType

model =  ModelFactory.create(
    model_platform=ModelPlatformType.OPENAI,
    model_type=ModelType.GPT_4O,
    model_config_dict={"temperature": 0.0},
)

