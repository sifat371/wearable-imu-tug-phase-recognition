from __future__ import annotations

import re
from typing import Optional

IMU_CHANNELS = [
    "LowerBack_Acc_X",
    "LowerBack_Acc_Y",
    "LowerBack_Acc_Z",
    "LowerBack_FreeAcc_E",
    "LowerBack_FreeAcc_N",
    "LowerBack_FreeAcc_U",
    "LowerBack_Gyr_X",
    "LowerBack_Gyr_Y",
    "LowerBack_Gyr_Z",
    "L_DorsalFoot_Acc_X",
    "L_DorsalFoot_Acc_Y",
    "L_DorsalFoot_Acc_Z",
    "L_DorsalFoot_FreeAcc_E",
    "L_DorsalFoot_FreeAcc_N",
    "L_DorsalFoot_FreeAcc_U",
    "L_DorsalFoot_Gyr_X",
    "L_DorsalFoot_Gyr_Y",
    "L_DorsalFoot_Gyr_Z",
    "R_DorsalFoot_Acc_X",
    "R_DorsalFoot_Acc_Y",
    "R_DorsalFoot_Acc_Z",
    "R_DorsalFoot_FreeAcc_E",
    "R_DorsalFoot_FreeAcc_N",
    "R_DorsalFoot_FreeAcc_U",
    "R_DorsalFoot_Gyr_X",
    "R_DorsalFoot_Gyr_Y",
    "R_DorsalFoot_Gyr_Z",
]

TUG_PHASES = ["Sitting", "SitToStand", "Walk", "Turn", "TurnToSit"]
SPMT_PHASES = ["Standing", "Walk", "Turn"]

TUG_TO_SPMT = {
    0: 0,
    1: 0,
    2: 1,
    3: 2,
    4: 0,
}

_TUG_LABEL_ALIASES = {
    "sitting": 0,
    "sit": 0,
    "sittostand": 1,
    "rising": 1,
    "standup": 1,
    "walk": 2,
    "walking": 2,
    "turn": 3,
    "turning": 3,
    "turntosit": 4,
    "standtosit": 4,
    "sittingdown": 4,
    "sitdown": 4,
}


def normalize_token(value: object) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "", str(value).strip().lower())


def map_tug_event(value: object) -> Optional[int]:
    return _TUG_LABEL_ALIASES.get(normalize_token(value))


def map_spmt_event(value: object) -> Optional[int]:
    token = normalize_token(value)
    if token in {"standing", "stand"}:
        return 0
    if token in {"walk", "walking"}:
        return 1
    if token in {"turn", "turning", "turnleft", "turnright"}:
        return 2
    return None


FOGSTAR_RAW_ACTIVITY_NAMES = {
    0: "Unknown",
    1: "Walk",
    2: "Sit",
    3: "Stand",
    4: "SitToStand",
    5: "StandToSit",
    6: "TurnRight",
    7: "TurnLeft",
}


def fogstar_target_definition(mode: str) -> tuple[list[str], dict[int, int], dict[int, int]]:
    """Return target phases, raw-activity mapping, and TUG projection."""
    if mode == "activity6":
        phases = ["Walk", "Sit", "Stand", "SitToStand", "StandToSit", "Turn"]
        raw_to_target = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 5}
        tug_to_target = {0: 1, 1: 3, 2: 0, 3: 5, 4: 4}
        return phases, raw_to_target, tug_to_target
    if mode == "coarse3":
        phases = ["StationaryTransition", "Walk", "Turn"]
        raw_to_target = {1: 1, 2: 0, 3: 0, 4: 0, 5: 0, 6: 2, 7: 2}
        tug_to_target = {0: 0, 1: 0, 2: 1, 3: 2, 4: 0}
        return phases, raw_to_target, tug_to_target
    raise ValueError(f"Unsupported FoG-STAR target mode: {mode}")
