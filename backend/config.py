import os
from dotenv import load_dotenv

load_dotenv()

RIOT_API_KEY = os.getenv("RIOT_API_KEY")

REGION_TO_PLATFORM = {
    'euw': 'euw1',
    'eune': 'eun1',
    'na': 'na1',
    'kr': 'kr',
    'jp': 'jp1',
    'br': 'br1',
    'lan': 'la1',
    'las': 'la2',
    'oce': 'oc1',
    'tr': 'tr1',
    'ru': 'ru',
}

PLATFORM_TO_REGIONAL = {
    'euw1': 'europe',
    'eun1': 'europe',
    'tr1': 'europe',
    'ru': 'europe',
    'na1': 'americas',
    'br1': 'americas',
    'la1': 'americas',
    'la2': 'americas',
    'oc1': 'americas',
    'kr': 'asia',
    'jp1': 'asia',
}

MODELS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'Training', 'models',
)
