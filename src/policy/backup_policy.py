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

    def should_return_to_depot(self, uav: UAV) -> bool:
        """Check if UAV should immediately return to depot."""
        # Critical: health depleted
        if uav.health <= 0.0:
            return True

        # Mission-complete condition: high revenue + poor link
        if (
            uav.collected_revenue >= self.config.mdp.action.return_.threshold.revenue
            and uav.link_quality <= self.config.mdp.action.return_.threshold.link_quality
        ):
            return True

        return False

    def should_backup(self, uav: UAV) -> bool:
        """Check if UAV should perform backup action."""
        score = self.compute_state_score(uav)
        return score < self.config.mdp.action.backup.threshold

    def decide_action(self, uav: UAV) -> str:
        """
        Decide the next action for the UAV based on its current state.
        
        Decision priority:
        1. Return to depot if health critical or mission-complete condition met
        2. Backup if overall state score is below threshold
        3. Continue otherwise
        
        Returns:
            "return_to_depot", "backup", or "continue"
        """
        if self.should_return_to_depot(uav):
            return "return_to_depot"

        if self.should_backup(uav):
            return "backup"

        return "continue"
