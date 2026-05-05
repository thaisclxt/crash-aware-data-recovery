from dataclasses import dataclass, asdict


@dataclass
class Metrics:
    # Policy decisions
    total_actions: int = 0
    backup_actions: int = 0
    continue_actions: int = 0

    # Mission outcomes
    crashes: int = 0
    completed_missions: int = 0 #  counts each time a UAV successfully returns to depot after finishing all its cycles (final mission success)
    successful_backups: int = 0
    failed_backups: int = 0

    # Value and risk
    total_revenue_collected: float = 0.0
    total_revenue_backed_up: float = 0.0
    total_risk_accumulated: int = 0

    def record_action(self, action: str) -> None:
        self.total_actions += 1

        if action == "backup":
            self.backup_actions += 1
        elif action == "continue":
            self.continue_actions += 1
        else:
            raise ValueError(f"Unknown action: {action}")

    def record_crash(self) -> None:
        self.crashes += 1

    def record_completed_mission(self) -> None:
        self.completed_missions += 1

    def record_backup(self, revenue: float, success: bool) -> None:
        if success:
            self.successful_backups += 1
            self.total_revenue_backed_up += revenue
        else:
            self.failed_backups += 1

    def record_risk_accumulated(self, risk: int) -> None:
        self.total_risk_accumulated += risk

    def record_revenue_collected(self, revenue: float) -> None:
        self.total_revenue_collected += revenue

    def to_dict(self) -> dict:
        return asdict(self)
