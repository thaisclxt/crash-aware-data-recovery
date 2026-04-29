from dataclasses import dataclass

from ..config import Config

from .UAV import UAV


@dataclass
class BackupPolicy:
    config: Config

    def score(self, uav: UAV) -> float:
        return (
            self.config.mdp.state.health.weight * uav.health
            + self.config.mdp.state.link_quality.weight * uav.link_quality
            + self.config.mdp.state.collected_revenue.weight * uav.collected_revenue
        )

    def decide_action(self, uav: UAV) -> str:
        score = self.score(uav)

        if uav.health == 0.0:
            return "return_to_depot"

        if (
            uav.collected_revenue >= self.config.mdp.action.return_.threshold.revenue
            and uav.link_quality <= self.config.mdp.action.return_.threshold.link_quality
        ):
            return "return_to_depot"

        if score < self.config.mdp.action.backup.threshold:
            return "backup"

        return "continue"
