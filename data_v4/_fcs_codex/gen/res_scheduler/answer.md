```python
import math


class AdaptiveSpotScheduler:
    def __init__(
        self,
        deadline: int,
        work: int,
        restart_overhead: int,
        spot_price: float,
        ondemand_price: float,
    ):
        self.deadline = int(deadline)
        self.work = int(work)
        self.restart_overhead = max(0, int(restart_overhead))
        self.spot_price = float(spot_price)
        self.ondemand_price = float(ondemand_price)

    def decide(self, t: int, spot_available: bool, progress: float) -> str:
        remaining = max(0.0, self.work - float(progress))
        remaining_units = int(math.ceil(remaining - 1e-12))
        time_left = self.deadline - int(t)

        if remaining_units <= 0:
            if spot_available and self.spot_price < self.ondemand_price:
                return "spot"
            return "ondemand"

        if time_left <= 0:
            return "ondemand"

        if remaining_units >= time_left:
            return "ondemand"

        if spot_available and self.spot_price < self.ondemand_price:
            return "spot"

        return "ondemand"
```