from enum import Enum

class BUBBLE_TYPE(Enum):
    """
    Enum representing different economic / tech bubble types.

    Each enum member has:
    - a human-readable name (AI, CRYPTO, etc.)
    - an associated string value used in file paths, datasets, etc.
    """

    AI = "ai"
    CRYPTO = "crypto"
    EV = "ev"
    DOTCOM = "dotcom"

    def dataset_filename(self) -> str:
        """Maps enum to dataset file name"""
        return f"{self.value}_merged.csv"

    @classmethod
    def from_str(cls, value: str) -> "BUBBLE_TYPE":
        return cls(value.lower())