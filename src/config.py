import random, yaml

from dataclasses import dataclass
from pathlib import Path


@dataclass
class SimulationConfig:
    time_limit: int
    seed: int
    number_runs: int
    generate_random_targets: bool

    def __post_init__(self) -> None:
        random.seed(self.seed)


@dataclass
class EnvironmentConfig:
    total_uavs: int
    total_targets: int
    depot_location: tuple[float, float]


@dataclass
class GridConfig:
    width: int
    height: int
    spacing: float


@dataclass
class UAVConfig:
    speed: float
    max_flight_time: float
    hover_time: float
    communication_range: float


@dataclass
class RevenueConfig:
    base: float
    min: float
    max: float
    update_interval: float


@dataclass
class RiskConfig:
    base: float
    min: float
    max: float
    update_interval: float


@dataclass
class WaypointConfig:
    revenue: RevenueConfig
    risk: RiskConfig


@dataclass
class HealthThresholdConfig:
    good: float
    warning: float


@dataclass
class HealthStateConfig:
    default: float
    alpha: float
    beta: float
    weight: float
    threshold: HealthThresholdConfig


@dataclass
class LinkQualityStateConfig:
    default: float
    weight: float


@dataclass
class CollectedRevenueThresholdConfig:
    low: float
    medium: float


@dataclass
class CollectedRevenueStateConfig:
    default: float
    weight: float
    threshold: CollectedRevenueThresholdConfig


@dataclass
class MDPStateConfig:
    health: HealthStateConfig
    link_quality: LinkQualityStateConfig
    collected_revenue: CollectedRevenueStateConfig


@dataclass
class ReturnActionThresholdConfig:
    link_quality: float
    revenue: float


@dataclass
class ReturnActionConfig:
    threshold: ReturnActionThresholdConfig


@dataclass
class BackupActionConfig:
    threshold: float


@dataclass
class MDPActionConfig:
    return_: ReturnActionConfig
    backup: BackupActionConfig


@dataclass
class MDPConfig:
    state: MDPStateConfig
    action: MDPActionConfig


@dataclass
class Config:
    simulation: SimulationConfig
    environment: EnvironmentConfig
    grid: GridConfig
    uav: UAVConfig
    waypoint: WaypointConfig
    mdp: MDPConfig


def load_configuration(path: Path) -> Config:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    simulation = data.get("simulation", {})
    environment = data.get("environment", {})
    grid = data.get("grid", {})
    uav = data.get("uav", {})
    waypoint = data.get("waypoint", {})
    mdp = data.get("mdp", {})

    waypoint_revenue = waypoint.get("revenue", {})
    waypoint_risk = waypoint.get("risk", {})

    mdp_state = mdp.get("state", {})
    mdp_action = mdp.get("action", {})

    health = mdp_state.get("health", {})
    health_threshold = health.get("threshold", {})

    link_quality = mdp_state.get("link_quality", {})

    collected_revenue = mdp_state.get("collected_revenue", {})
    collected_revenue_threshold = collected_revenue.get("threshold", {})

    return_action = mdp_action.get("return", {})
    return_threshold = return_action.get("threshold", {})

    backup_action = mdp_action.get("backup", {})

    depot_location = tuple(environment.get("depot_location", (0.0, 0.0)))

    return Config(
        simulation=SimulationConfig(
            time_limit=simulation.get("time_limit"),
            seed=simulation.get("seed"),
            number_runs=simulation.get("number_runs"),
            generate_random_targets=simulation.get("generate_random_targets"),
        ),
        environment=EnvironmentConfig(
            total_uavs=environment.get("total_uavs"),
            total_targets=environment.get("total_targets"),
            depot_location=depot_location,
        ),
        grid=GridConfig(
            width=grid.get("width"),
            height=grid.get("height"),
            spacing=grid.get("spacing"),
        ),
        uav=UAVConfig(
            speed=uav.get("speed"),
            max_flight_time=uav.get("max_flight_time"),
            hover_time=uav.get("hover_time"),
            communication_range=uav.get("communication_range"),
        ),
        waypoint=WaypointConfig(
            revenue=RevenueConfig(
                base=waypoint_revenue.get("base"),
                min=waypoint_revenue.get("min"),
                max=waypoint_revenue.get("max"),
                update_interval=waypoint_revenue.get("update_interval"),
            ),
            risk=RiskConfig(
                base=waypoint_risk.get("base"),
                min=waypoint_risk.get("min"),
                max=waypoint_risk.get("max"),
                update_interval=waypoint_risk.get("update_interval"),
            ),
        ),
        mdp=MDPConfig(
            state=MDPStateConfig(
                health=HealthStateConfig(
                    default=health.get("default"),
                    alpha=health.get("alpha"),
                    beta=health.get("beta"),
                    weight=health.get("weight"),
                    threshold=HealthThresholdConfig(
                        good=health_threshold.get("good"),
                        warning=health_threshold.get("warning"),
                    ),
                ),
                link_quality=LinkQualityStateConfig(
                    default=link_quality.get("default"),
                    weight=link_quality.get("weight"),
                ),
                collected_revenue=CollectedRevenueStateConfig(
                    default=collected_revenue.get("default"),
                    weight=collected_revenue.get("weight"),
                    threshold=CollectedRevenueThresholdConfig(
                        low=collected_revenue_threshold.get("low"),
                        medium=collected_revenue_threshold.get("medium"),
                    ),
                ),
            ),
            action=MDPActionConfig(
                return_=ReturnActionConfig(
                    threshold=ReturnActionThresholdConfig(
                        link_quality=return_threshold.get("link_quality"),
                        revenue=return_threshold.get("revenue"),
                    ),
                ),
                backup=BackupActionConfig(
                    threshold=backup_action.get("threshold"),
                ),
            ),
        ),
    )
