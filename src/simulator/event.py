from dataclasses import dataclass, field
from typing import Optional

@dataclass(order=True)
class Event:
    time: float
    priority: int
    event_type: str = field(compare=False)
    uav_id: Optional[int] = field(default=None, compare=False)
