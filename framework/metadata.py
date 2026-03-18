"""Species and domain metadata for the DCASE2026 mosquito baseline.

Author: Yuanbo Hou
Email: Yuanbo.Hou@eng.ox.ac.uk
Affiliation: Machine Learning Research Group, University of Oxford
"""

import re
from pathlib import Path
from typing import List, Tuple, Union


SPECIES_ID_TO_NAME = {
    "1": "Aedes aegypti",
    "2": "Aedes albopictus",
    "3": "Culex quinquefasciatus",
    "4": "Anopheles gambiae",
    "5": "Anopheles arabiensis",
    "6": "Anopheles dirus",
    "7": "Culex pipiens",
    "8": "Anopheles minimus",
    "9": "Anopheles stephensi",
}
SPECIES_NAMES = list(SPECIES_ID_TO_NAME.values())
SPECIES_TO_INDEX = {name: idx for idx, name in enumerate(SPECIES_NAMES)}
DOMAIN_ID_TO_NAME = {"1": "D1", "2": "D2", "3": "D3", "4": "D4", "5": "D5"}
DOMAIN_NAMES = list(DOMAIN_ID_TO_NAME.values())
DOMAIN_TO_INDEX = {name: idx for idx, name in enumerate(DOMAIN_NAMES)}
FILE_PATTERN = re.compile(r"^S_(\d+)_D_(\d+)_(\d+)$")


def load_id_list(path: Union[str, Path]) -> List[str]:
    with open(path, "r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def parse_file_id(file_id: str) -> Tuple[str, str]:
    match = FILE_PATTERN.match(file_id)
    if not match:
        raise ValueError(f"Unrecognized file id: {file_id}")
    species_id, domain_id, _ = match.groups()
    if species_id not in SPECIES_ID_TO_NAME:
        raise ValueError(f"Unknown species id in file id: {file_id}")
    if domain_id not in DOMAIN_ID_TO_NAME:
        raise ValueError(f"Unknown domain id in file id: {file_id}")
    return SPECIES_ID_TO_NAME[species_id], DOMAIN_ID_TO_NAME[domain_id]
