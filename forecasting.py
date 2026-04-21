"""
forecasting.py — PiezoPrev Final
Previsão recursiva frequência-consciente com suporte a cenários.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from training import check_extrapolation
from config import HORIZON_RELIABILITY_MSGS, HORIZON_RELIABILITY_MSG_GENERIC, DEFAULT_FEATURES

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Conversão horizonte dias → passos de leitura
# ---------------------------------------------------------------------------

def horizon_days_to_steps(n_days: int, freq_hours: float) -> int:
    """
    Converte horizonte em dias para número de passos de previsão.
    Para instrumento automático (4h): 30 dias × 6 leituras/dia = 180 passos.
    Para instrumento diário: 30 dias × 1 leitura/dia = 30 passos.
    """
    readings_per_day = max(1, round(24.0 / freq_hours))
    return max(1, n_days * readings_per_day)


# ---------------------------------------------------------------------------
# Construção do vetor de features para um passo futuro
# ---------------------------------------------------------------------------

def _build_step_features(
    working_df: pd.DataFrame,
    feature_cols: list,
    target_col: str,
    group_col: str,
    instrumento: str,
    step_date: pd.Timestamp,
    ext_row: Optional[dict],
    cfg,
) -> pd.DataFrame:
    """
    Reconstrói features para o próximo passo de previsão,
    usando o histórico acumulado (com previsões anteriores).
    """
    from features import (
        create_lags, create_rolling_stats, create_deltas,
        create_rain_features, create_reservoir_features,
        create_calendar_features, create_permeability_features,
    )

    last_row = working_df.iloc[-1].copy()
    next_row = last_row.copy()
    next_row["data"] = step_date

    # Variáveis externas: cenário ou último valor observado
    for col in ["pluviometria", "nivel_reservatorio"]:
        if ext_row and col in ext_row:
            next_row[col] = ext_row[col]

    next_df = pd.concat(
        [working_df, pd.DataFrame([next_row])],
        ignore_index=True,
    ).sort_values("data").reset_index(drop=True)

    # Recalcular features para a nova linha
    next_df = create_lags(next_df, target_col, cfg.lag_list, group_col)
    next_df = create_rolling_stats(next_df, target_col, cfg.rolling_windows, group_col)
    next_df = create_deltas(next_df, target_col, cfg.delta_periods, group_col)
    next_df = create_rain_features(next_df, "pluviometria", cfg.rain_windows, group_col)
    next_df = create_reservoir_features(
        next_df, "nivel_reservatorio", cfg.reservoir_delta_windows, group_col
    )
    next_df = create_calendar_features(next_df, "data")
    next_df = create_permeability_features(next_df)

    step_row = next_df.iloc[[-1]].copy()
    for col in feature_cols:
        if col not in step_row.columns:
            step_row[col] = 0.0

    return step_row[feature_cols].fillna(0)


# ---------------------------------------------------------------------------
# Forecasting recursivo
# ---------------------------------------------------------------------------

def recursive_forecast(
    model: XGBRegressor,
    df_history: pd.DataFrame,
    feature_cols: list,
    instrumento: str,
    n_days: int = 30,
    freq_hours: float = 24.0,
    target_col: str = "leitura",
    group_col: str = "instrumento",
    date_col: str = "data",
    cfg=None,
    scenario: Optional[pd.DataFrame] = None,
    X_train: Optional[pd.DataFrame] = None,
) -> dict:
    """
    Gera previsões futuras recursivas respeitando a frequência do instrumento.

    Args:
        model: Modelo treinado.
        df_history: Histórico completo com features.
        feature_cols: Features do modelo.
        instrumento: Nome do instrumento a prever.
        n_days: Horizonte desejado em DIAS.
        freq_hours: Frequência do instrumento em horas.
        target_col: Coluna alvo.
        group_col: Coluna de instrumento.
        date_col: Coluna de data.
        cfg: FeatureConfig. Se None, usa DEFAULT_FEATURES.
        scenario: DataFrame com variáveis externas futuras (opcional).
        X_train: Features de treino para alertas de extrapolação.

    Returns:
        Dict com: forecast_df, n_steps, freq_hours, mode,
                  extrapolation_alerts, reliability_warnings
    """
    if cfg is None:
        cfg = DEFAULT_FEATURES

    mode = "cenário" if scenario is not None else "automático"
    reliability_warnings = []

    # Aviso de confiabilidade baseado no horizonte
    rel_msg = HORIZON_RELIABILITY_MSGS.get(n_days)
    if rel_msg is None and n_days > 30:
        # Para valores não mapeados (ex: 180 dias), usa a msg genérica
        rel_msg = HORIZON_RELIABILITY_MSG_GENERIC
    if rel_msg:
        reliability_warnings.append(rel_msg)

    n_steps = horizon_days_to_steps(n_days, freq_hours)
    readings_per_day = max(1, round(24.0 / freq_hours))
    freq_label = f"{freq_hours:.1f}h"

    logger.info(
        f"Forecast: instrumento={instrumento} | {n_days} dias = "
        f"{n_steps} leituras | freq={freq_label}"
    )

    # Filtrar histórico do instrumento
    hist = df_history[df_history[group_col] == instrumento].copy()
    hist = hist.sort_values(date_col).reset_index(drop=True)

    if len(hist) < 10:
        raise ValueError(
            f"Histórico insuficiente para '{instrumento}': {len(hist)} registros. "
            "Mínimo recomendado: 10."
        )

    freq_td   = pd.Timedelta(hours=freq_hours)
    last_date = hist[date_col].max()

    # Índice do cenário de entrada
    scenario_index = {}
    if scenario is not None:
        scenario = scenario.copy()
        scenario[date_col] = pd.to_datetime(scenario[date_col], errors="coerce")
        for _, row in scenario.iterrows():
            scenario_index[row[date_col]] = row.to_dict()

    predictions = []
    working_df  = hist.copy()

    for step in range(1, n_steps + 1):
        next_date       = last_date + freq_td * step
        dia_equivalente = round(step / max(readings_per_day, 1), 1)

        # Obter valores externos do cenário para este passo
        ext_row = scenario_index.get(next_date, None)
        if ext_row is None and scenario is not None:
            past_dates = [d for d in scenario_index if d <= next_date]
            if past_dates:
                ext_row = scenario_index[max(past_dates)]

        try:
            X_step = _build_step_features(
                working_df, feature_cols, target_col, group_col,
                instrumento, next_date, ext_row, cfg,
            )
        except Exception as e:
            logger.warning(f"Erro ao construir features no passo {step}: {e}")
            X_step = pd.DataFrame([{col: 0.0 for col in feature_cols}])

        pred_val = float(model.predict(X_step)[0])

        # Atualizar histórico de trabalho com previsão
        new_row               = working_df.iloc[-1].copy()
        new_row[date_col]     = next_date
        new_row[target_col]   = pred_val
        working_df = pd.concat(
            [working_df, pd.DataFrame([new_row])],
            ignore_index=True,
        )

        predictions.append({
            "data":            next_date,
            "previsao":        pred_val,
            "passo":           step,
            "dia_equivalente": dia_equivalente,
            "modo":            mode,
        })

    forecast_df = pd.DataFrame(predictions)

    # Verificar extrapolação
    extrapolation_alerts = []
    if X_train is not None and not forecast_df.empty:
        try:
            last_row = working_df.iloc[[-1]]
            avail    = [c for c in feature_cols if c in last_row.columns]
            if avail:
                extrapolation_alerts = check_extrapolation(
                    X_train, last_row[avail], avail
                )
        except Exception:
            pass

    return {
        "forecast_df":          forecast_df,
        "n_steps":              n_steps,
        "freq_hours":           freq_hours,
        "mode":                 mode,
        "extrapolation_alerts": extrapolation_alerts,
        "reliability_warnings": reliability_warnings,
    }


# ---------------------------------------------------------------------------
# Predição pontual manual
# ---------------------------------------------------------------------------

def predict_manual(
    model: XGBRegressor,
    feature_cols: list,
    values: dict,
    X_train: Optional[pd.DataFrame] = None,
) -> dict:
    """Previsão pontual a partir de valores informados manualmente."""
    row     = {col: float(values.get(col, 0.0)) for col in feature_cols}
    X_input = pd.DataFrame([row])
    pred_val = float(model.predict(X_input)[0])

    alerts = []
    if X_train is not None:
        alerts = check_extrapolation(X_train, X_input, feature_cols)

    return {
        "previsao":       round(pred_val, 4),
        "features_usadas": row,
        "alertas":        alerts,
    }
