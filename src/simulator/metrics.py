from dataclasses import dataclass, asdict


@dataclass
class Metrics:
    total_crashes: int = 0
    total_completed_missions: int = 0
    total_backups: int = 0
    total_lost_revenue: float = 0.0
    total_risk_accumulated: float = 0.0
    total_revenue_collected: float = 0.0
    total_revenue_backed_up: float = 0.0
    backup_actions: int = 0
    continue_actions: int = 0
    successful_backups: int = 0
    failed_backups: int = 0


    def record_action(self, action: str) -> None:
        if action == "backup":
            self.backup_actions += 1
        elif action == "continue":
            self.continue_actions += 1
        else:
            raise ValueError(f"Unknown action: {action}")


    def record_crash(self) -> None:
        self.total_crashes += 1


    def record_completed_mission(self) -> None:
        self.total_completed_missions += 1


    def record_backup(self, revenue: float, success: bool) -> None:
        if success:
            self.total_backups += 1
            self.successful_backups += 1
            self.total_revenue_backed_up += revenue
        else:
            self.failed_backups += 1


    def record_lost_revenue(self, amount: float) -> None:
        self.total_lost_revenue += amount


    def record_risk_accumulated(self, amount: float) -> None:
        self.total_risk_accumulated += amount


    def record_revenue_collected(self, amount: float) -> None:
        self.total_revenue_collected += amount


    def to_dict(self) -> dict:
        return asdict(self)
