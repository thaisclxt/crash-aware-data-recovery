from ..config import Config
from ..models.uav import UAV


class BackupPolicy:
    def __init__(self, config: Config) -> None:
        self.backup_treshold = config.mdp.action.backup.threshold
        
        self.health_weight = config.mdp.state.health.weight
        self.link_weight = config.mdp.state.link_quality.weight
        self.rev_weight = config.mdp.state.collected_revenue.weight
    

    def _compute_score(self, uav: UAV) -> float:
        """
        Compute a weighted state score where a higher score means better peformance.
        """
        return (
            self.health_weight * uav.health
            + self.link_weight * uav.link_quality
            + self.rev_weight * uav.collected_revenue
        )


    def _should_backup(self, uav: UAV) -> bool:
        return self._compute_score(uav) > self.backup_treshold
    

    def decide_action(self, uav: UAV) -> str:
        if self._should_backup(uav):
            return "backup"
        return "continue"
