from ..config import Config
from ..models.uav import UAV


class BackupPolicy:
    def __init__(self, config: Config) -> None:
        self.config = config
    

    def compute_state_score(self, uav: UAV) -> float:
        """Compute a weighted state quality score (higher = better)."""
        return (
            self.config.mdp.state.health.weight * uav.health
            + self.config.mdp.state.link_quality.weight * uav.link_quality
            + self.config.mdp.state.collected_revenue.weight * uav.collected_revenue
        )


    def should_backup(self, uav: UAV) -> bool:
        """Check if UAV should perform backup action."""
        score = self.compute_state_score(uav)
        return score < self.config.mdp.action.backup.threshold
    

    def decide_action(self, uav: UAV) -> str:
        if self.should_backup(uav):
            return "backup"

        return "continue"
