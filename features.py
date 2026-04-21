"""
features.py — PiezoPrev Final (corrigido)

Correções aplicadas nesta versão:
─────────────────────────────────────────────────────────────────────────────
1. build_features()
   - Log diagnóstico com contagem de linhas ANTES e DEPOIS de cada etapa
   - dropna() restrito apenas ao lag_1 e target (não a todas as colunas)
   - Aviso quando mais de 30% das linhas são perdidas no dropna de lags
   - Não remove linhas por NaN em features opcionais (rain, reservoir, etc.)

2. create_instrument_identity_features()
   - Protegido contra séries com < 5 leituras (não crashar, só pular)

3. create_rain_features() / create_reservoir_features()
   - Retornam sem modificar o df se a coluna não existir (comportamento já ok)
   - Não introduzem NaN desnecessários com min_periods muito alto

4. Política geral de NaN após feature engineering:
   - O único dropna() obrigatório é no lag_1 (precisa de pelo menos 1 leitura anterior)
   - Features de rain/reservoir ausentes são preenchidas com 0 (não removem linhas)
   - Features de identidade com min_periods reduzido para séries curtas
─────────────────────────────────────────────────────────────────────────────
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

from config import DEFAULT_FEATURES, FeatureConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lags
# ---------------------------------------------------------------------------

def create_lags(df, col, lags, group_col="instrumento", prefix=None):
    df = df.copy()
    pfx = prefix or col
    grouped = df.groupby(group_col, sort=False)[col]
    for lag in lags:
        df[f"{pfx}_lag_{lag}"] = grouped.shift(lag)
    return df


# ---------------------------------------------------------------------------
# Médias e desvios móveis (sem leakage)
# ---------------------------------------------------------------------------

def create_rolling_stats(df, col, windows, group_col="instrumento", prefix=None):
    df = df.copy()
    pfx = prefix or col
    for w in windows:
        shifted = df.groupby(group_col, sort=False)[col].shift(1)
        # min_periods = metade da janela para não zerar séries curtas
        mp = max(1, w // 2)
        df[f"{pfx}_rmean_{w}"] = (
            shifted.groupby(df[group_col])
            .transform(lambda x: x.rolling(w, min_periods=mp).mean())
        )
        df[f"{pfx}_rstd_{w}"] = (
            shifted.groupby(df[group_col])
            .transform(lambda x: x.rolling(w, min_periods=mp).std().fillna(0))
        )
    return df


# ---------------------------------------------------------------------------
# Deltas
# ---------------------------------------------------------------------------

def create_deltas(df, col, periods, group_col="instrumento", prefix=None):
    df = df.copy()
    pfx = prefix or col
    grouped = df.groupby(group_col, sort=False)[col]
    for p in periods:
        df[f"{pfx}_delta_{p}"] = grouped.shift(1) - grouped.shift(1 + p)
    return df


# ---------------------------------------------------------------------------
# Acumulados de chuva
# ---------------------------------------------------------------------------

def create_rain_features(df, rain_col="pluviometria", windows=None, group_col="instrumento"):
    """
    Cria acumulados de chuva. Se a coluna não existir, retorna df sem modificar.
    NaN resultantes são preenchidos com 0 (ausência de chuva = sem chuva registrada).
    Isso evita que linhas sem dado de pluviometria sejam removidas no dropna().
    """
    if rain_col not in df.columns:
        return df
    if windows is None:
        windows = [3, 7, 15, 30]

    df = df.copy()
    for w in windows:
        mp = max(1, w // 4)   # min_periods menor para séries curtas
        shifted = df.groupby(group_col, sort=False)[rain_col].shift(1)
        col_name = f"chuva_acum_{w}p"
        df[col_name] = (
            shifted.groupby(df[group_col])
            .transform(lambda x: x.rolling(w, min_periods=mp).sum())
        )
        # Preencher NaN restantes com 0 — não remove linhas
        df[col_name] = df[col_name].fillna(0)

    shifted = df.groupby(group_col, sort=False)[rain_col].shift(1)
    max_w = max(windows) if windows else 7
    col_max = f"chuva_max_{max_w}p"
    df[col_max] = (
        shifted.groupby(df[group_col])
        .transform(lambda x: x.rolling(max_w, min_periods=1).max())
    ).fillna(0)

    return df


# ---------------------------------------------------------------------------
# Variações do reservatório
# ---------------------------------------------------------------------------

def create_reservoir_features(
    df, reservoir_col="nivel_reservatorio", delta_windows=None, group_col="instrumento"
):
    """
    Cria variações do nível do reservatório. Se a coluna não existir, retorna sem modificar.
    NaN preenchidos com 0 para não remover linhas no dropna posterior.
    """
    if reservoir_col not in df.columns:
        return df
    if delta_windows is None:
        delta_windows = [1, 7]

    df = df.copy()
    grouped = df.groupby(group_col, sort=False)[reservoir_col]

    for lag in [1, 7]:
        df[f"reserv_lag_{lag}"] = grouped.shift(lag).fillna(method="bfill").fillna(0)

    for w in delta_windows:
        col_name = f"reserv_delta_{w}p"
        df[col_name] = (grouped.shift(1) - grouped.shift(1 + w)).fillna(0)

    w_roll = delta_windows[-1] if delta_windows else 7
    shifted = grouped.shift(1)
    df[f"reserv_rmean_{w_roll}"] = (
        shifted.groupby(df[group_col])
        .transform(lambda x: x.rolling(w_roll, min_periods=1).mean())
    ).fillna(method="bfill").fillna(0)

    return df


# ---------------------------------------------------------------------------
# Features de calendário
# ---------------------------------------------------------------------------

def create_calendar_features(df, date_col="data"):
    df = df.copy()
    dt = df[date_col].dt
    df["cal_dia_semana"] = dt.dayofweek
    df["cal_dia_mes"]    = dt.day
    df["cal_mes"]        = dt.month
    df["cal_trimestre"]  = dt.quarter
    df["cal_dia_ano"]    = dt.dayofyear
    df["cal_hora"]       = dt.hour

    df["cal_mes_sin"]  = np.sin(2 * np.pi * df["cal_mes"] / 12)
    df["cal_mes_cos"]  = np.cos(2 * np.pi * df["cal_mes"] / 12)
    df["cal_doy_sin"]  = np.sin(2 * np.pi * df["cal_dia_ano"] / 365.25)
    df["cal_doy_cos"]  = np.cos(2 * np.pi * df["cal_dia_ano"] / 365.25)
    df["cal_hora_sin"] = np.sin(2 * np.pi * df["cal_hora"] / 24)
    df["cal_hora_cos"] = np.cos(2 * np.pi * df["cal_hora"] / 24)
    return df


# ---------------------------------------------------------------------------
# Permeabilidade
# ---------------------------------------------------------------------------

def create_permeability_features(df):
    if "permeabilidade" not in df.columns:
        return df
    df = df.copy()
    perm = df["permeabilidade"].copy()
    df["permeab_log10"] = np.where(perm > 0, np.log10(perm.clip(lower=1e-15)), np.nan)
    # Preenche com mediana para não gerar NaN desnecessário
    median_val = df["permeab_log10"].median()
    df["permeab_log10"] = df["permeab_log10"].fillna(median_val if not np.isnan(median_val) else 0)
    return df


# ---------------------------------------------------------------------------
# Features de identidade do instrumento
# ---------------------------------------------------------------------------

def create_instrument_identity_features(
    df: pd.DataFrame,
    target_col: str = "leitura",
    group_col: str = "instrumento",
    date_col: str = "data",
) -> pd.DataFrame:
    """
    Cria features que capturam a individualidade de cada instrumento
    com base exclusivamente na sua própria série histórica.

    Ajuste desta versão:
    - min_periods reduzido para funcionar com séries curtas
    - Instrumentos com < 5 leituras recebem 0 nas features de identidade
      em vez de NaN (evita remoção por dropna posterior)
    - fillna(0) em todas as features de identidade ao final
    """
    df = df.copy()

    for inst, grp in df.groupby(group_col, sort=False):
        idx    = grp.index
        series = df.loc[idx, target_col]
        n_valid = int(series.notna().sum())

        # Mínimo para calcular qualquer estatística
        if n_valid < 5:
            for col in [
                "inst_mean_expanding", "inst_std_expanding",
                "inst_zscore_local", "inst_pct_rank",
                "inst_rain_corr_30", "inst_trend_30",
            ]:
                df.loc[idx, col] = 0.0
            continue

        shifted = series.shift(1)

        # min_periods menor para séries curtas
        mp_expand = min(3, max(1, n_valid // 10))

        expanding_mean = shifted.expanding(min_periods=mp_expand).mean()
        expanding_std  = shifted.expanding(min_periods=mp_expand).std().fillna(1e-9)

        df.loc[idx, "inst_mean_expanding"] = expanding_mean
        df.loc[idx, "inst_std_expanding"]  = expanding_std
        df.loc[idx, "inst_zscore_local"]   = (
            (shifted - expanding_mean) / expanding_std.replace(0, 1e-9)
        )
        df.loc[idx, "inst_pct_rank"] = shifted.expanding(min_periods=mp_expand).rank(pct=True)

        if "pluviometria" in df.columns:
            shifted_rain = df.loc[idx, "pluviometria"].shift(1)
            mp_corr = min(10, max(3, n_valid // 5))
            rolling_corr = (
                shifted.rolling(30, min_periods=mp_corr).corr(shifted_rain).fillna(0)
            )
            df.loc[idx, "inst_rain_corr_30"] = rolling_corr
        else:
            df.loc[idx, "inst_rain_corr_30"] = 0.0

        def local_slope(x):
            if x.isna().all() or len(x.dropna()) < 3:
                return 0.0
            valid = x.dropna()
            t = np.arange(len(valid))
            if t.std() == 0:
                return 0.0
            try:
                return float(np.polyfit(t, valid.values, 1)[0])
            except Exception:
                return 0.0

        mp_trend = min(10, max(3, n_valid // 5))
        trend = shifted.rolling(30, min_periods=mp_trend).apply(local_slope, raw=False)
        df.loc[idx, "inst_trend_30"] = trend

    # Preencher NaN restantes com 0 — evita perda de linhas no dropna posterior
    identity_cols = [
        "inst_mean_expanding", "inst_std_expanding",
        "inst_zscore_local", "inst_pct_rank",
        "inst_rain_corr_30", "inst_trend_30",
    ]
    for col in identity_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    return df


# ---------------------------------------------------------------------------
# Pipeline completo — com diagnóstico de linhas por etapa
# ---------------------------------------------------------------------------

def build_features(
    df: pd.DataFrame,
    target_col: str = "leitura",
    date_col: str = "data",
    group_col: str = "instrumento",
    cfg: FeatureConfig = DEFAULT_FEATURES,
    drop_na_lags: bool = True,
) -> pd.DataFrame:
    """
    Pipeline completo de feature engineering com diagnóstico de linhas por etapa.

    Política de NaN desta versão:
    - O único dropna() obrigatório é no lag_1 + target (precisa de 1 leitura anterior)
    - Features opcionais (rain, reservoir, identity) têm NaN preenchidos com 0
    - Isso preserva o máximo de linhas para o modelo
    """
    logger.info(
        f"=== Feature engineering | freq={cfg.freq_hours:.1f}h | "
        f"lags={cfg.lag_list} | rain={cfg.rain_windows} ==="
    )
    n_in = len(df)
    logger.info(f"[FE] Entrada: {n_in} linhas")

    df = create_lags(df, target_col, cfg.lag_list, group_col)
    logger.info(f"[FE] Após lags: {len(df)} linhas")

    df = create_rolling_stats(df, target_col, cfg.rolling_windows, group_col)
    logger.info(f"[FE] Após rolling stats: {len(df)} linhas")

    df = create_deltas(df, target_col, cfg.delta_periods, group_col)
    logger.info(f"[FE] Após deltas: {len(df)} linhas")

    df = create_rain_features(df, "pluviometria", cfg.rain_windows, group_col)
    logger.info(f"[FE] Após rain features: {len(df)} linhas")

    df = create_reservoir_features(
        df, "nivel_reservatorio", cfg.reservoir_delta_windows, group_col
    )
    logger.info(f"[FE] Após reservoir features: {len(df)} linhas")

    df = create_calendar_features(df, date_col)
    logger.info(f"[FE] Após calendar features: {len(df)} linhas")

    df = create_permeability_features(df)
    logger.info(f"[FE] Após permeability features: {len(df)} linhas")

    df = create_instrument_identity_features(df, target_col, group_col, date_col)
    logger.info(f"[FE] Após identity features: {len(df)} linhas")

    # ── dropna RESTRITO ao lag_1 e target ──────────────────────────────────
    # Não remove linhas por NaN em features opcionais.
    # lag_1 é NaN somente na primeira linha de cada instrumento (inevitável).
    if drop_na_lags:
        essential_lag = f"{target_col}_lag_1"
        if essential_lag in df.columns:
            n_before = len(df)
            df = df.dropna(subset=[essential_lag, target_col])
            dropped = n_before - len(df)
            if dropped > 0:
                pct_dropped = 100 * dropped / max(n_before, 1)
                logger.info(
                    f"[FE] dropna(lag_1 + target): {dropped} linhas removidas "
                    f"({pct_dropped:.1f}%) → {len(df)} restantes"
                )
                if pct_dropped > 30:
                    logger.warning(
                        f"[FE] ATENÇÃO: {pct_dropped:.1f}% das linhas foram removidas "
                        f"pelo dropna de lags. Isso pode indicar:\n"
                        f"  • Lags longos demais para o tamanho da série\n"
                        f"  • Merge malsucedido gerando muitos NaN no target\n"
                        f"  • Séries muito curtas por instrumento\n"
                        f"  Linhas antes: {n_before} | Linhas depois: {len(df)}"
                    )

    logger.info(
        f"=== Feature engineering concluído: {n_in} → {len(df)} linhas | "
        f"{df.shape[1]} colunas ==="
    )

    if len(df) == 0:
        logger.error(
            "[FE] RESULTADO VAZIO após feature engineering! "
            "O modelo não poderá ser treinado. "
            "Verifique os logs acima para identificar em qual etapa as linhas foram perdidas."
        )

    return df


# ---------------------------------------------------------------------------
# Seleção de features
# ---------------------------------------------------------------------------

def get_feature_columns(
    df: pd.DataFrame,
    target_col: str = "leitura",
    group_col: str = "instrumento",
    exclude_extra: Optional[list] = None,
) -> list:
    always_exclude = {
        "data", group_col, target_col,
        "tipo_instrumento", "tipo_material",
        "outlier_iqr", "outlier_zscore", "outlier_jump",
        "outlier_flag", "outlier_reason",
    }
    if exclude_extra:
        always_exclude.update(exclude_extra)

    feature_cols = sorted([
        col for col in df.columns
        if col not in always_exclude
        and df[col].dtype not in [object]
        and not df[col].isna().all()
    ])
    logger.info(f"Features selecionadas: {len(feature_cols)}")
    return feature_cols
