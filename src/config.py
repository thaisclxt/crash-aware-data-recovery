import random, yaml

from dataclasses import dataclass
from pathlib import Path


@dataclass
class SimulationConfig:
    time_limit: int
    seed: int
    generate_random_targets: bool

    def __post_init__(self) -> None:
        random.seed(self.seed)


@dataclass
class EnvironmentConfig:
    total_uavs: list
    total_targets: int
    depot_location: tuple[float, float]
    fixed_targets: list[tuple[float, float]]


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
    preparation_time: float
    communication_range: float
    base_crash_probability: list


@dataclass
class RevenueConfig:
    min: float
    max: float
    update_interval: float


@dataclass
class RiskConfig:
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
class BackedUpRevenueStateConfig:
    default: float


@dataclass
class MDPStateConfig:
    health: HealthStateConfig
    link_quality: LinkQualityStateConfig
    collected_revenue: CollectedRevenueStateConfig
    backed_up_revenue: BackedUpRevenueStateConfig


@dataclass
class BackupActionConfig:
    threshold: float


@dataclass
class MDPActionConfig:
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

    backed_up_revenue = mdp_state.get("backed_up_revenue", {})

    return_action = mdp_action.get("return", {})
    return_threshold = return_action.get("threshold", {})

    backup_action = mdp_action.get("backup", {})

    return Config(
        simulation=SimulationConfig(
            time_limit=simulation.get("time_limit"),
            seed=simulation.get("seed"),
            generate_random_targets=simulation.get("generate_random_targets"),
        ),
        environment=EnvironmentConfig(
            total_uavs=environment.get("total_uavs"),
            total_targets=environment.get("total_targets"),
            depot_location=tuple(environment.get("depot_location", (0.0, 0.0))),
            fixed_targets=environment.get("fixed_targets", []),
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
            preparation_time=uav.get("preparation_time"),
            communication_range=uav.get("communication_range"),
            base_crash_probability=uav.get("base_crash_probability"),
        ),
        waypoint=WaypointConfig(
            revenue=RevenueConfig(
                min=waypoint_revenue.get("min"),
                max=waypoint_revenue.get("max"),
                update_interval=waypoint_revenue.get("update_interval"),
            ),
            risk=RiskConfig(
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
                backed_up_revenue=BackedUpRevenueStateConfig(
                    default=backed_up_revenue.get("default"),
                )
            ),
            action=MDPActionConfig(
                backup=BackupActionConfig(
                    threshold=backup_action.get("threshold"),
                ),
            ),
        ),
    )
