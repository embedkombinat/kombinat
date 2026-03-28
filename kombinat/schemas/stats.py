from pydantic import BaseModel


class StatsOut(BaseModel):
    total_pairs: int
    unlabeled_pairs: int
    verified_pairs: int
    rejected_pairs: int
    active_contributors_24h: int
    total_contributors: int
    pairs_per_day: int
    total_input_tokens: int
    total_output_tokens: int
