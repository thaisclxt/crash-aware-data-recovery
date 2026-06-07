from dataclasses import dataclass, asdict


@dataclass
class Metrics:
    completed_tours: int = 0
    total_crashes: int = 0
    total_lost_revenue: float = 0.0
    total_revenue_backed_up: float = 0.0
    total_delivered_revenue: float = 0.0
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


    def record_backup(self, revenue: float, success: bool) -> None:
        if success:
            self.successful_backups += 1
            self.total_revenue_backed_up += revenue
        else:
            self.failed_backups += 1


    def record_crash(self) -> None:
        self.total_crashes += 1


    def record_completed_tours(self) -> None:
        self.completed_tours += 1


    def record_lost_revenue(self, amount: float) -> None:
        self.total_lost_revenue += amount

    
    def record_delivered_revenue(self, amount: float) -> None:
        self.total_delivered_revenue += amount


    def to_dict(self) -> dict:
        return asdict(self)
