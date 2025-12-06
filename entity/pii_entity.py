from typing import Optional


class PiiEntity:
    name: str
    value: str
    position: Optional[tuple]


def create_pii(name: str, val: str, start: int = None, end: int = None):
    pii = PiiEntity()
    pii.name = name
    pii.value = val
    if start is not None and end is not None:
        pii.position = (start, end)

