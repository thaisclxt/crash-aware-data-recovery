from dataclasses import dataclass, asdict


@dataclass
class Metrics:
    # Policy decisions
    total_actions: int = 0
    backup_actions: int = 0
    return_actions: int = 0
    continue_actions: int = 0

    # Mission outcomes
    crashes: int = 0
    completed_sequences: int = 0
    depot_deliveries: int = 0
    successful_backups: int = 0
    failed_backups: int = 0

    # Value and risk
    total_revenue_collected: float = 0.0
    revenue_delivered_to_depot: float = 0.0
    total_risk_accumulated: float = 0.0

    def record_action(self, action: str) -> None:
        self.total_actions += 1

        if action == "backup":
            self.backup_actions += 1
        elif action == "return_to_depot":
            self.return_actions += 1
        elif action == "continue":
            self.continue_actions += 1
        else:
            raise ValueError(f"Unknown action: {action}")

    def record_crash(self) -> None:
        self.crashes += 1

    def record_completed_sequence(self) -> None:
        self.completed_sequences += 1

    def record_depot_delivery(self, revenue: float) -> None:
        self.depot_deliveries += 1
        self.revenue_delivered_to_depot += revenue

    def record_backup(self, success: bool) -> None:
        if success:
            self.successful_backups += 1
        else:
            self.failed_backups += 1

    def add_revenue_collected(self, revenue: float) -> None:
        self.total_revenue_collected += revenue

    def add_risk_accumulated(self, risk: float) -> None:
        self.total_risk_accumulated += risk

    def to_dict(self) -> dict:
        return asdict(self)
