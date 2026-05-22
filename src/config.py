from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class PipelineConfig:
    train_path: Path = Path("ecommerce_price_prediction-train.csv")
    test_path: Path = Path("ecommerce_price_prediction-test-3-days.csv")
    output_dir: Path = Path("outputs")
    target: str = "price"
    date_col: str = "capturedAt"
    random_seed: int = 42
    validation_days: int = 3
    anchors_per_day: int = 100
    entity_smoothing: float = 20.0
    calibration_smoothing: float = 8.0
    calibration_delta_cap: float | None = None
    selective_calibration: bool = False
    calibration_min_anchors: int = 20
    calibration_max_residual_iqr: float | None = None
    calibration_min_abs_global_delta: float = 0.0
    second_stage_alpha: float = 10.0
    include_experimental_variants: bool = False
    prediction_variant: str = "hybrid_last_price_entity"
    catboost_iterations: int = 1200
    catboost_learning_rate: float = 0.05
    catboost_depth: int = 8
    id_cols: list[str] = field(default_factory=lambda: ["shopId", "itemId", "modelId"])
    cat_cols: list[str] = field(
        default_factory=lambda: [
            "shopId",
            "itemId",
            "modelId",
            "promotionId",
            "cat_id",
            "brand",
        ]
    )
    bool_cols: list[str] = field(
        default_factory=lambda: [
            "is_free_shipping",
            "is_pre_order",
            "is_official_shop",
            "is_verified",
            "is_preferred_plus_seller",
        ]
    )
    history_keys: list[str] = field(
        default_factory=lambda: ["modelId", "itemId", "shopId", "cat_id", "brand"]
    )
    calibration_keys: list[str] = field(
        default_factory=lambda: ["cat_id", "shopId", "brand", "promotionId"]
    )
