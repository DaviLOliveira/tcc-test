"""
config.py — PiezoPrev Final
Configurações centralizadas do sistema.

ATENÇÃO: A base de permeabilidade NÃO está mais hardcoded aqui.
Ela é lida de arquivo externo via permeability_db.py.
Este arquivo contém apenas constantes de esquema, frequência, horizontes e
parâmetros de modelagem.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import pandas as pd


# ---------------------------------------------------------------------------
# Tipos de instrumento e frequências
# ---------------------------------------------------------------------------

class InstrumentType(str, Enum):
    PIEZ_MANUAL = "Piezômetro Manual"
    PIEZ_AUTO   = "Piezômetro Automático"
    MNA_MANUAL  = "Medidor de Nível d'Água Manual"
    MNA_AUTO    = "Medidor de Nível d'Água Automatizado"


INSTRUMENT_FREQ_HOURS: dict = {
    InstrumentType.PIEZ_MANUAL: 24.0,
    InstrumentType.PIEZ_AUTO:   4.0,
    InstrumentType.MNA_MANUAL:  24.0,
    InstrumentType.MNA_AUTO:    4.0,
}

INSTRUMENT_FREQ_LABEL: dict = {
    InstrumentType.PIEZ_MANUAL: "Manual (~1×/dia)",
    InstrumentType.PIEZ_AUTO:   "Automático (~6×/dia, 4h)",
    InstrumentType.MNA_MANUAL:  "Manual (~1×/dia)",
    InstrumentType.MNA_AUTO:    "Automático (~6×/dia, 4h)",
}


# ---------------------------------------------------------------------------
# FeatureConfig dependente de frequência
# ---------------------------------------------------------------------------

@dataclass
class FeatureConfig:
    """
    Configuração das features em NÚMERO DE PERÍODOS (não em dias absolutos).
    Para 4h: lag_1=4h, lag_6=~1dia, lag_42=~7dias.
    Para 24h: lag_1=1dia, lag_7=7dias.
    """
    lag_list: list               = field(default_factory=lambda: [1, 3, 7, 14])
    rolling_windows: list        = field(default_factory=lambda: [3, 7, 14])
    delta_periods: list          = field(default_factory=lambda: [1, 3, 7])
    rain_windows: list           = field(default_factory=lambda: [3, 7, 15, 30])
    reservoir_delta_windows: list = field(default_factory=lambda: [1, 7])
    freq_hours: float            = 24.0


def detect_series_frequency(series_dates: "pd.Series") -> dict:
    """
    Infere a frequência predominante de uma série temporal de forma robusta.

    Estratégia:
    1. Calcula intervalos entre leituras consecutivas
    2. Filtra lacunas > 30 dias e intervalos < 1 min (duplicatas)
    3. Arredonda para valores candidatos e calcula a moda
    4. Usa moda se cobertura >= 60%, senão usa mediana
    5. Retorna confiança: alta / média / baixa

    Returns:
        Dict: {freq_hours, median_hours, mode_hours, confidence, n_valid_intervals}
    """
    dates = pd.to_datetime(series_dates, errors="coerce").dropna().sort_values()
    result = {
        "freq_hours": 24.0,
        "median_hours": 24.0,
        "mode_hours": 24.0,
        "confidence": "baixa",
        "n_valid_intervals": 0,
    }

    if len(dates) < 5:
        return result

    deltas_h = dates.diff().dropna().dt.total_seconds() / 3600.0
    valid = deltas_h[(deltas_h >= (1 / 60)) & (deltas_h <= 24 * 30)]
    result["n_valid_intervals"] = int(len(valid))

    if len(valid) < 3:
        return result

    median_h = float(valid.median())
    result["median_hours"] = round(median_h, 2)

    candidates = [0.25, 0.5, 1, 2, 4, 6, 8, 12, 24, 48, 72, 168]
    rounded = valid.apply(lambda x: min(candidates, key=lambda c: abs(c - x)))
    mode_h = float(rounded.mode().iloc[0]) if not rounded.empty else median_h
    result["mode_hours"] = mode_h

    mode_coverage = float((rounded == mode_h).sum() / len(valid))

    if mode_coverage >= 0.60:
        freq = mode_h
        confidence = "alta" if mode_coverage >= 0.80 else "média"
    else:
        freq = median_h
        confidence = "baixa"

    result["freq_hours"] = max(0.25, freq)
    result["confidence"] = confidence
    return result


def get_feature_config_for_instrument(
    inst_type: InstrumentType,
    detected_freq_hours: Optional[float] = None,
) -> FeatureConfig:
    """Retorna FeatureConfig escalado à frequência real do instrumento."""
    base_freq = INSTRUMENT_FREQ_HOURS[inst_type]
    freq = detected_freq_hours if (detected_freq_hours and detected_freq_hours > 0) else base_freq
    factor = max(1, round(24.0 / freq))

    if factor == 1:
        return FeatureConfig(
            lag_list=[1, 3, 7, 14],
            rolling_windows=[3, 7, 14],
            delta_periods=[1, 3, 7],
            rain_windows=[3, 7, 15, 30],
            reservoir_delta_windows=[1, 7],
            freq_hours=freq,
        )
    return FeatureConfig(
        lag_list=[1, factor, factor * 3, factor * 7],
        rolling_windows=[factor, factor * 3, factor * 7],
        delta_periods=[1, factor, factor * 3],
        rain_windows=[factor, factor * 3, factor * 7, factor * 15],
        reservoir_delta_windows=[factor, factor * 7],
        freq_hours=freq,
    )


# ---------------------------------------------------------------------------
# Esquema de dados
# ---------------------------------------------------------------------------

REQUIRED_COLUMNS: list  = ["data", "instrumento", "leitura"]
BUSINESS_REQUIRED: list = ["pluviometria", "tipo_material", "permeabilidade"]
OPTIONAL_COLUMNS: list  = ["nivel_reservatorio", "profundidade", "cota_instalacao"]
INSTRUMENT_METADATA_COLS: list = [
    "tipo_instrumento", "tipo_material", "permeabilidade", "profundidade", "cota_instalacao"
]
EXTERNAL_TIME_COLS: list = ["pluviometria", "nivel_reservatorio"]
CATEGORICAL_COLS: list   = ["tipo_instrumento", "tipo_material"]

# Mapeamento de nomes amigáveis de colunas → nomes internos
COLUMN_ALIASES: dict = {
    "codigo do instrumento": "instrumento",
    "código do instrumento": "instrumento",
    "cod instrumento": "instrumento",
    "cod_instrumento": "instrumento",
    "id instrumento": "instrumento",
    "id_instrumento": "instrumento",
    "instrumento": "instrumento",
    "data da leitura": "data",
    "data leitura": "data",
    "data_leitura": "data",
    "date": "data",
    "timestamp": "data",
    "valor": "leitura",
    "leitura": "leitura",
    "nivel": "leitura",
    "nível": "leitura",
    "cota": "leitura",
    "reading": "leitura",
    "chuva": "pluviometria",
    "chuva (mm)": "pluviometria",
    "precipitacao": "pluviometria",
    "precipitação": "pluviometria",
    "rain": "pluviometria",
    "rainfall": "pluviometria",
    "pluviometria": "pluviometria",
    "na": "nivel_reservatorio",
    "nivel da agua": "nivel_reservatorio",
    "nível da água": "nivel_reservatorio",
    "nivel reservatorio": "nivel_reservatorio",
    "nível reservatório": "nivel_reservatorio",
    "cota reservatorio": "nivel_reservatorio",
    "nivel_reservatorio": "nivel_reservatorio",
}


# ---------------------------------------------------------------------------
# Horizontes de previsão
# ---------------------------------------------------------------------------

# (label, n_days, alerta_longo)
FORECAST_HORIZONS_PRODUCAO: list = [
    ("30 dias",  30,  False),
    ("90 dias",  90,  False),
    ("180 dias", 180, True),
]

FORECAST_HORIZONS_EXPERIMENTAL: list = [
    ("1 ano (experimental)", 365, True),
]

FORECAST_DEFAULT_LABEL = "30 dias"

# Mensagens de confiabilidade por horizonte em dias
HORIZON_RELIABILITY_MSGS: dict = {
    30:  None,
    90:  (
        "ℹ️ Horizonte de 90 dias: confiabilidade moderada. "
        "O erro acumulado do forecast recursivo pode ser relevante "
        "a partir da segunda metade do período."
    ),
    180: (
        "⚠️ Horizonte de 180 dias: confiabilidade reduzida. "
        "O erro se propaga a cada passo recursivo. "
        "Interprete como tendência direcional, não como valor preciso."
    ),
    365: (
        "🔴 MODO EXPERIMENTAL — 1 ANO: confiabilidade muito baixa. "
        "A previsão recursiva acumula erro significativo ao longo de 365 passos. "
        "Use apenas para análise exploratória de tendência de longo prazo. "
        "NÃO utilize para decisões operacionais ou de segurança de barragens."
    ),
}

# Mensagem genérica para compatibilidade (usada em forecasting.py)
HORIZON_RELIABILITY_MSG_GENERIC = (
    "⚠️ Horizonte longo: o erro acumula a cada passo recursivo. "
    "Interprete como tendência qualitativa, não como valor preciso."
)


# ---------------------------------------------------------------------------
# Configuração de hyperparameter tuning (Optuna)
# ---------------------------------------------------------------------------

@dataclass
class TuningConfig:
    enabled: bool        = True
    n_trials: int        = 25
    timeout_seconds: int = 120
    cv_folds: int        = 3
    metric: str          = "rmse"
    verbosity: int       = 0


# ---------------------------------------------------------------------------
# Configuração de validação cruzada temporal
# ---------------------------------------------------------------------------

@dataclass
class CVConfig:
    n_splits: int      = 5
    test_size: int     = None
    gap: int           = 0


# ---------------------------------------------------------------------------
# Parâmetros de qualidade e modelagem
# ---------------------------------------------------------------------------

@dataclass
class MissingValueConfig:
    max_gap_target: int     = 5
    max_gap_external: int   = 3
    long_gap_threshold: int = 10


@dataclass
class OutlierConfig:
    iqr_factor: float           = 3.0
    zscore_window: int          = 30
    zscore_threshold: float     = 4.0
    max_consecutive_jump: float = 5.0


@dataclass
class ModelConfig:
    n_estimators: int          = 500
    learning_rate: float       = 0.05
    max_depth: int             = 6
    min_child_weight: int      = 5
    subsample: float           = 0.8
    colsample_bytree: float    = 0.8
    reg_alpha: float           = 0.1
    reg_lambda: float          = 1.0
    early_stopping_rounds: int = 40
    random_state: int          = 42
    n_jobs: int                = -1
    eval_metric: str           = "rmse"
    min_test_samples: int      = 20
    test_ratio: float          = 0.2


MODELING_STRATEGY: str = "global_with_instrument_identity"

DEFAULT_MISSING  = MissingValueConfig()
DEFAULT_OUTLIER  = OutlierConfig()
DEFAULT_FEATURES = FeatureConfig()
DEFAULT_MODEL    = ModelConfig()
DEFAULT_TUNING   = TuningConfig()
DEFAULT_CV       = CVConfig()
