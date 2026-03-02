from pydantic import BaseModel, field_validator
from backend.config import REGION_TO_PLATFORM


class PredictRequest(BaseModel):
    game_name: str
    tag_line: str
    region: str = 'euw'

    @field_validator('game_name')
    @classmethod
    def validate_game_name(cls, v):
        v = v.strip()
        if not v or len(v) > 24:
            raise ValueError('Invalid game name')
        return v

    @field_validator('tag_line')
    @classmethod
    def validate_tag_line(cls, v):
        v = v.strip()
        if not v or len(v) > 5:
            raise ValueError('Invalid tag line')
        return v

    @field_validator('region')
    @classmethod
    def validate_region(cls, v):
        v = v.strip().lower()
        if v not in REGION_TO_PLATFORM:
            raise ValueError(f'Invalid region. Valid: {list(REGION_TO_PLATFORM.keys())}')
        return v
