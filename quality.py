"""
quality.py
----------
Controle de qualidade dos dados: tratamento criterioso de valores ausentes
e detecção/sinalização de outliers.

Filosofia:
- Não interpolar automaticamente lacunas longas (mascararia eventos reais).
- Distinguir tipos de lacuna e registrar rastreabilidade.
- Sinalizar outliers sem removê-los cegamente.
- Toda operação gera um log de auditoria (QualityReport).
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import logging

import numpy as np
import pandas as pd

from config import (
    DEFAULT_MISSING,
    DEFAULT_OUTLIER,
    MissingValueConfig,
    OutlierConfig,
    CATEGORICAL_COLS,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Relatório de qualidade
# ---------------------------------------------------------------------------

@dataclass
class QualityReport:
    """Rastreabilidade das operações de qualidade."""
    missing_summary: dict = field(default_factory=dict)   # {instrumento: {col: contagens}}
    outlier_summary: dict = field(default_factory=dict)   # {instrumento: n_flagged}
    interpolated_counts: dict = field(default_factory=dict)
    long_gaps: dict = field(default_factory=dict)         # {instrumento: lista de lacunas longas}
    warnings: list[str] = field(default_factory=list)

    def to_dataframe(self) -> pd.DataFrame:
        """Converte o resumo de ausentes em DataFrame para exibição."""
        rows = []
        for inst, cols in self.missing_summary.items():
            for col, info in cols.items():
                rows.append({"instrumento": inst, "coluna": col, **info})
        return pd.DataFrame(rows) if rows else pd.DataFrame()


# ---------------------------------------------------------------------------
# Detecção de lacunas
# ---------------------------------------------------------------------------

def _classify_gaps(
    series: pd.Series,
    cfg: MissingValueConfig,
) -> pd.DataFrame:
    """
    Classifica lacunas em uma série temporal.

    Tipos:
    - "curta": até max_gap_target períodos consecutivos → interpolar
    - "longa": acima de long_gap_threshold → não interpolar, manter NaN
    - "media": entre os dois → interpolar com cautela e registrar

    Returns:
        DataFrame com colunas: start_idx, end_idx, length, type
    """
    mask = series.isna()
    if not mask.any():
        return pd.DataFrame(columns=["start_idx", "end_idx", "length", "gap_type"])

    gaps = []
    in_gap = False
    start = None

    for i, is_null in enumerate(mask):
        if is_null and not in_gap:
            in_gap = True
            start = i
        elif not is_null and in_gap:
            in_gap = False
            length = i - start
            if length <= cfg.max_gap_target:
                gap_type = "curta"
            elif length >= cfg.long_gap_threshold:
                gap_type = "longa"
            else:
                gap_type = "media"
            gaps.append({"start_idx": start, "end_idx": i - 1, "length": length, "gap_type": gap_type})

    # Lacuna no fim da série
    if in_gap:
        length = len(mask) - start
        gap_type = "curta" if length <= cfg.max_gap_target else ("longa" if length >= cfg.long_gap_threshold else "media")
        gaps.append({"start_idx": start, "end_idx": len(mask) - 1, "length": length, "gap_type": gap_type})

    return pd.DataFrame(gaps)


# ---------------------------------------------------------------------------
# Tratamento de valores ausentes
# ---------------------------------------------------------------------------

def treat_missing_values(
    df: pd.DataFrame,
    target_col: str = "leitura",
    group_col: str = "instrumento",
    cfg: MissingValueConfig = DEFAULT_MISSING,
) -> tuple[pd.DataFrame, QualityReport]:
    """
    Trata valores ausentes de forma criteriosa por instrumento.

    Regras:
    - Variável alvo (leitura):
        * Lacunas CURTAS (≤ max_gap_target): interpolação linear
        * Lacunas MÉDIAS: interpolação com aviso registrado
        * Lacunas LONGAS (≥ long_gap_threshold): mantidas como NaN
          (serão removidas na etapa de feature engineering)
    - Variáveis externas numéricas (pluviometria, nível_reservatório):
        * Forward fill limitado (max_gap_external)
        * Preenchimento com 0 apenas para pluviometria (ausência = sem chuva)
    - Variáveis categóricas: forward fill dentro do instrumento
    - Metadados do instrumento (permeabilidade, profundidade, etc.):
        * Forward fill; se ausente em toda a série, mantido como NaN (não substituir por 0)

    Args:
        df: DataFrame ordenado por instrumento e data.
        target_col: Coluna alvo.
        group_col: Coluna de agrupamento.
        cfg: Configuração de valores ausentes.

    Returns:
        Tuple (DataFrame tratado, QualityReport).
    """
    df = df.copy()
    report = QualityReport()

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in CATEGORICAL_COLS if c in df.columns]

    # Colunas externas que representam acumulações (zerar quando ausente é razoável)
    zero_fill_cols = {"pluviometria"}

    # Colunas de metadados do instrumento (valores fixos no tempo, nunca zerar)
    metadata_cols = {"permeabilidade", "profundidade", "cota_instalacao"}

    for inst, group in df.groupby(group_col, sort=False):
        idx = group.index
        report.missing_summary[inst] = {}
        report.long_gaps[inst] = []

        # --- Variável alvo ---
        if target_col in df.columns:
            series = df.loc[idx, target_col].copy()
            n_missing_before = series.isna().sum()
            gaps_df = _classify_gaps(series, cfg)

            for _, gap in gaps_df.iterrows():
                abs_start = idx[gap["start_idx"]]
                abs_end = idx[gap["end_idx"]]

                if gap["gap_type"] in ("curta", "media"):
                    # Interpola os índices desse gap específico
                    df.loc[abs_start:abs_end, target_col] = (
                        df.loc[idx, target_col].interpolate(method="linear", limit=cfg.max_gap_target)
                        .loc[abs_start:abs_end]
                    )
                    if gap["gap_type"] == "media":
                        report.warnings.append(
                            f"[{inst}] '{target_col}': lacuna média de {gap['length']} períodos "
                            f"interpolada (índices {gap['start_idx']}→{gap['end_idx']})."
                        )
                else:
                    report.long_gaps[inst].append({
                        "coluna": target_col,
                        "inicio_idx": int(gap["start_idx"]),
                        "fim_idx": int(gap["end_idx"]),
                        "comprimento": int(gap["length"]),
                    })

            n_missing_after = df.loc[idx, target_col].isna().sum()
            n_interpolated = n_missing_before - n_missing_after
            report.missing_summary[inst][target_col] = {
                "ausentes_originais": int(n_missing_before),
                "interpolados": int(n_interpolated),
                "ausentes_restantes": int(n_missing_after),
            }
            if n_interpolated > 0:
                report.interpolated_counts[inst] = report.interpolated_counts.get(inst, 0) + n_interpolated

        # --- Variáveis numéricas externas ---
        for col in numeric_cols:
            if col == target_col:
                continue
            series = df.loc[idx, col]
            n_null = series.isna().sum()
            if n_null == 0:
                continue

            if col in metadata_cols:
                # Forward fill apenas — não zerar metadados
                filled = series.fillna(method="ffill")
                df.loc[idx, col] = filled
            elif col in zero_fill_cols:
                # Forward fill limitado, depois zera (ausência de chuva = 0)
                filled = series.fillna(method="ffill", limit=cfg.max_gap_external).fillna(0)
                df.loc[idx, col] = filled
            else:
                # Forward fill limitado para outras externas
                filled = series.fillna(method="ffill", limit=cfg.max_gap_external)
                df.loc[idx, col] = filled

            report.missing_summary[inst].setdefault(col, {
                "ausentes_originais": int(n_null),
                "interpolados": 0,
                "ausentes_restantes": int(df.loc[idx, col].isna().sum()),
            })

        # --- Variáveis categóricas ---
        for col in cat_cols:
            if col in df.columns:
                df.loc[idx, col] = (
                    df.loc[idx, col]
                    .fillna(method="ffill")
                    .fillna(method="bfill")
                    .fillna("Desconhecido")
                )

    return df, report


# ---------------------------------------------------------------------------
# Detecção de outliers
# ---------------------------------------------------------------------------

def detect_outliers(
    df: pd.DataFrame,
    target_col: str = "leitura",
    group_col: str = "instrumento",
    cfg: OutlierConfig = DEFAULT_OUTLIER,
) -> tuple[pd.DataFrame, dict]:
    """
    Detecta outliers e anomalias na variável alvo, sem removê-los.

    Estratégia dupla:
    1. IQR por instrumento: marca valores além de Q1 - k*IQR e Q3 + k*IQR
    2. Z-score móvel: marca desvios abruptos em janela temporal

    Adiciona ao DataFrame:
    - `outlier_iqr`: bool — anomalia por IQR estático
    - `outlier_zscore`: bool — anomalia por z-score móvel
    - `outlier_jump`: bool — salto abrupto > N desvios-padrão
    - `outlier_flag`: bool — OR de todos os critérios
    - `outlier_reason`: str — motivo(s) do flag

    Args:
        df: DataFrame com dados de leituras.
        target_col: Coluna alvo.
        group_col: Coluna de instrumento.
        cfg: Configuração de outliers.

    Returns:
        Tuple (DataFrame com colunas de outlier, dict de resumo por instrumento).
    """
    df = df.copy()
    summary = {}

    # Inicializar colunas
    for col in ["outlier_iqr", "outlier_zscore", "outlier_jump", "outlier_flag"]:
        df[col] = False
    df["outlier_reason"] = ""

    for inst, group in df.groupby(group_col, sort=False):
        idx = group.index
        series = df.loc[idx, target_col].copy()

        # Ignorar instrumentos com dados insuficientes
        n_valid = series.dropna().shape[0]
        if n_valid < 10:
            continue

        reasons = pd.Series("", index=idx)

        # --- 1. IQR ---
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1

        if iqr > 0:
            lower = q1 - cfg.iqr_factor * iqr
            upper = q3 + cfg.iqr_factor * iqr
            mask_iqr = (series < lower) | (series > upper)
            df.loc[idx[mask_iqr], "outlier_iqr"] = True
            reasons[idx[mask_iqr]] += "IQR;"

        # --- 2. Z-score móvel ---
        roll_mean = series.rolling(cfg.zscore_window, min_periods=5, center=False).mean().shift(1)
        roll_std = series.rolling(cfg.zscore_window, min_periods=5, center=False).std().shift(1)

        with np.errstate(invalid="ignore"):
            z_scores = np.abs((series - roll_mean) / roll_std.replace(0, np.nan))

        mask_z = z_scores > cfg.zscore_threshold
        mask_z = mask_z.fillna(False)
        df.loc[idx[mask_z], "outlier_zscore"] = True
        reasons[idx[mask_z]] += "ZSCORE;"

        # --- 3. Salto abrupto ---
        diff = series.diff().abs()
        std_global = series.std()
        if std_global > 0:
            mask_jump = diff > cfg.max_consecutive_jump * std_global
            mask_jump = mask_jump.fillna(False)
            df.loc[idx[mask_jump], "outlier_jump"] = True
            reasons[idx[mask_jump]] += "JUMP;"

        # --- Flag consolidado ---
        flag_mask = (
            df.loc[idx, "outlier_iqr"] |
            df.loc[idx, "outlier_zscore"] |
            df.loc[idx, "outlier_jump"]
        )
        df.loc[idx, "outlier_flag"] = flag_mask
        df.loc[idx, "outlier_reason"] = reasons.str.rstrip(";")

        n_flagged = int(flag_mask.sum())
        summary[inst] = {
            "n_total": n_valid,
            "n_outliers": n_flagged,
            "pct_outliers": round(100 * n_flagged / max(n_valid, 1), 2),
            "faixa_valida": (round(float(series.dropna().min()), 4),
                             round(float(series.dropna().max()), 4)),
        }

    logger.info(
        f"Detecção de outliers concluída: "
        f"{df['outlier_flag'].sum()} registros sinalizados de {len(df):,}"
    )
    return df, summary


def apply_outlier_filter(
    df: pd.DataFrame,
    mode: str = "flag_only",
    target_col: str = "leitura",
) -> pd.DataFrame:
    """
    Aplica o filtro de outliers conforme a estratégia escolhida.

    Modos:
    - "flag_only": Apenas mantém as colunas de flag, não altera os dados (recomendado)
    - "nullify": Substitui outliers por NaN (serão tratados como ausentes)
    - "remove": Remove registros sinalizados completamente

    Args:
        df: DataFrame com colunas de outlier.
        mode: Estratégia de aplicação.
        target_col: Coluna alvo.

    Returns:
        DataFrame tratado.
    """
    if "outlier_flag" not in df.columns:
        return df

    df = df.copy()
    n_flagged = df["outlier_flag"].sum()

    if mode == "flag_only":
        logger.info(f"Outliers mantidos como flags: {n_flagged} registros sinalizados.")

    elif mode == "nullify":
        df.loc[df["outlier_flag"], target_col] = np.nan
        logger.info(f"Outliers substituídos por NaN: {n_flagged} registros.")

    elif mode == "remove":
        df = df[~df["outlier_flag"]].reset_index(drop=True)
        logger.info(f"Outliers removidos: {n_flagged} registros.")

    else:
        raise ValueError(f"Modo de outlier inválido: {mode}. Use 'flag_only', 'nullify' ou 'remove'.")

    return df
