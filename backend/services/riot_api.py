import time
from urllib.parse import quote
import httpx
from backend.config import RIOT_API_KEY

_ddragon_version_cache = None
_champion_map_cache = None


class RiotAPIError(Exception):
    def __init__(self, message, status_code=None):
        super().__init__(message)
        self.status_code = status_code


class RiotAPIService:
    def __init__(self, region='euw'):
        from backend.config import REGION_TO_PLATFORM, PLATFORM_TO_REGIONAL
        self.platform = REGION_TO_PLATFORM.get(region, 'euw1')
        self.regional = PLATFORM_TO_REGIONAL.get(self.platform, 'europe')
        self.headers = {'X-Riot-Token': RIOT_API_KEY}

    def _request(self, url, retries=3):
        for attempt in range(retries):
            response = httpx.get(url, headers=self.headers, timeout=15.0)
            if response.status_code == 200:
                return response.json()
            if response.status_code == 429:
                retry_after = int(response.headers.get('Retry-After', 2))
                time.sleep(retry_after)
                continue
            if response.status_code == 404:
                raise RiotAPIError("Player not found", 404)
            if response.status_code == 403:
                raise RiotAPIError("API key expired or invalid", 403)
            raise RiotAPIError(f"Riot API error: {response.status_code}", response.status_code)
        raise RiotAPIError("Rate limit exceeded, try again later", 429)

    def get_account(self, game_name, tag_line):
        url = (
            f"https://{self.regional}.api.riotgames.com"
            f"/riot/account/v1/accounts/by-riot-id/{quote(game_name)}/{quote(tag_line)}"
        )
        return self._request(url)

    def get_ranked_match_ids(self, puuid, count=20):
        url = (
            f"https://{self.regional}.api.riotgames.com"
            f"/lol/match/v5/matches/by-puuid/{puuid}/ids"
            f"?queue=420&type=ranked&start=0&count={count}"
        )
        return self._request(url)

    def get_match(self, match_id):
        url = (
            f"https://{self.regional}.api.riotgames.com"
            f"/lol/match/v5/matches/{match_id}"
        )
        return self._request(url)

    def get_player_matches(self, game_name, tag_line, num_matches=10):
        account = self.get_account(game_name, tag_line)
        puuid = account['puuid']

        match_ids = self.get_ranked_match_ids(puuid, count=num_matches + 10)
        if not match_ids:
            raise RiotAPIError("No ranked matches found for this player", 404)

        matches = []
        for match_id in match_ids:
            if len(matches) >= num_matches:
                break
            try:
                match_data = self.get_match(match_id)
                participant = self._extract_participant(match_data, puuid)
                if participant:
                    matches.append(participant)
            except RiotAPIError:
                continue

        if len(matches) < 3:
            raise RiotAPIError(
                f"Not enough ranked matches found (found {len(matches)}, need at least 3)", 404,
            )

        return {
            'puuid': puuid,
            'game_name': account.get('gameName', game_name),
            'tag_line': account.get('tagLine', tag_line),
            'matches': matches,
        }

    def _extract_participant(self, match_data, puuid):
        info = match_data.get('info', {})

        if info.get('queueId') != 420:
            return None

        participants = info.get('participants', [])
        player = None
        for p in participants:
            if p.get('puuid') == puuid:
                player = p
                break
        if not player:
            return None

        time_played = player.get('timePlayed', info.get('gameDuration', 0))
        if time_played < 300:
            return None

        game_minutes = time_played / 60.0
        kills = player.get('kills', 0)
        deaths = player.get('deaths', 0)
        assists = player.get('assists', 0)
        total_minions = player.get('totalMinionsKilled', 0)
        neutral_minions = player.get('neutralMinionsKilled', 0)
        vision_score = player.get('visionScore', 0)
        damage_dealt = player.get('totalDamageDealtToChampions', 0)
        damage_taken = player.get('totalDamageTaken', 0)
        gold_earned = player.get('goldEarned', 0)
        win = 1 if player.get('win', False) else 0

        kda = (kills + assists) / max(1, deaths)
        cs_per_min = (total_minions + neutral_minions) / game_minutes
        damage_per_min = damage_dealt / game_minutes
        gold_per_min = gold_earned / game_minutes
        deaths_per_min = deaths / game_minutes
        vision_per_min = vision_score / game_minutes
        damage_taken_per_min = damage_taken / game_minutes

        return {
            'matchId': match_data.get('metadata', {}).get('matchId', ''),
            'championName': player.get('championName', ''),
            'role': player.get('teamPosition', 'UNKNOWN'),
            'kills': kills,
            'deaths': deaths,
            'assists': assists,
            'totalMinionsKilled': total_minions,
            'neutralMinionsKilled': neutral_minions,
            'visionScore': vision_score,
            'totalDamageDealtToChampions': damage_dealt,
            'totalDamageTaken': damage_taken,
            'goldEarned': gold_earned,
            'timePlayed': time_played,
            'gameDuration': info.get('gameDuration', time_played),
            'win': win,
            'kda': round(kda, 4),
            'cs_per_min': round(cs_per_min, 4),
            'damage_per_min': round(damage_per_min, 4),
            'gold_per_min': round(gold_per_min, 4),
            'deaths_per_min': round(deaths_per_min, 4),
            'vision_per_min': round(vision_per_min, 4),
            'damage_taken_per_min': round(damage_taken_per_min, 4),
        }

    def get_active_game(self, puuid):
        url = (
            f"https://{self.platform}.api.riotgames.com"
            f"/lol/spectator/v5/active-games/by-summoner/{puuid}"
        )
        try:
            return self._request(url)
        except RiotAPIError as e:
            if e.status_code == 404:
                return None
            raise

    def get_account_by_puuid(self, puuid):
        url = (
            f"https://{self.regional}.api.riotgames.com"
            f"/riot/account/v1/accounts/by-puuid/{puuid}"
        )
        return self._request(url)

    def get_ddragon_version(self):
        global _ddragon_version_cache
        if _ddragon_version_cache is None:
            resp = httpx.get(
                "https://ddragon.leagueoflegends.com/api/versions.json", timeout=10.0,
            )
            _ddragon_version_cache = resp.json()[0]
        return _ddragon_version_cache

    def get_champion_id_map(self):
        global _champion_map_cache
        if _champion_map_cache is None:
            version = self.get_ddragon_version()
            url = f"https://ddragon.leagueoflegends.com/cdn/{version}/data/en_US/champion.json"
            resp = httpx.get(url, timeout=15.0)
            data = resp.json()
            _champion_map_cache = {}
            for name, info in data['data'].items():
                _champion_map_cache[int(info['key'])] = info['id']
        return _champion_map_cache

    def get_player_matches_by_puuid(self, puuid, num_matches=5):
        try:
            match_ids = self.get_ranked_match_ids(puuid, count=num_matches + 5)
        except RiotAPIError:
            return None
        if not match_ids:
            return None

        matches = []
        for match_id in match_ids:
            if len(matches) >= num_matches:
                break
            try:
                match_data = self.get_match(match_id)
                participant = self._extract_participant(match_data, puuid)
                if participant:
                    matches.append(participant)
            except RiotAPIError:
                continue

        return matches if len(matches) >= 2 else None
