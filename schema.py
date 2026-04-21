"""
schema.py — PiezoPrev Final
Validação do esquema de dados de entrada.
"""

from dataclasses import dataclass, field
import pandas as pd
import numpy as np

from config import (
    REQUIRED_COLUMNS, BUSINESS_REQUIRED, OPTIONAL_COLUMNS,
    CATEGORICAL_COLS, EXTERNAL_TIME_COLS, detect_series_frequency,
)


@dataclass
class ValidationReport:
    ok: bool
    missing_required: list = field(default_factory=list)
    present_optional: list = field(default_factory=list)
    absent_optional: list  = field(default_factory=list)
    type_warnings: list    = field(default_factory=list)
    info_messages: list    = field(default_factory=list)
    n_rows: int            = 0
    n_instruments: int     = 0
    date_range: tuple      = ()
    detected_freq_hours: float = 24.0

    def summary(self) -> str:
        lines = [
            f"{'✅' if self.ok else '❌'} Validação do arquivo de leituras",
            f"  • Registros: {self.n_rows:,}",
            f"  • Instrumentos: {self.n_instruments}",
        ]
        if self.date_range:
            lines.append(f"  • Período: {self.date_range[0]} → {self.date_range[1]}")
        lines.append(f"  • Frequência detectada: ~{self.detected_freq_hours:.1f}h entre leituras")
        if self.missing_required:
            lines.append(f"  • ❌ Colunas obrigatórias ausentes: {self.missing_required}")
        if self.absent_optional:
            lines.append(f"  • ⚠️  Opcionais ausentes: {self.absent_optional}")
        if self.present_optional:
            lines.append(f"  • ✓  Opcionais presentes: {self.present_optional}")
        for w in self.type_warnings:
            lines.append(f"  • ⚠️  {w}")
        for m in self.info_messages:
            lines.append(f"  • ℹ️  {m}")
        return "\n".join(lines)


def validate_schema(df: pd.DataFrame, date_col: str = "data") -> ValidationReport:
    """
    Valida o arquivo de leituras do instrumento.
    Exige: data, instrumento, leitura.
    Detecta frequência temporal automaticamente.
    """
    report = ValidationReport(ok=True, n_rows=len(df))

    missing_req = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_req:
        report.ok = False
        report.missing_required = missing_req
        return report

    report.present_optional = [c for c in OPTIONAL_COLUMNS if c in df.columns]
    report.absent_optional  = [c for c in OPTIONAL_COLUMNS if c not in df.columns]

    # Datas
    try:
        parsed = pd.to_datetime(df[date_col], errors="coerce", dayfirst=True)
        n_invalid = int(parsed.isna().sum())
        if n_invalid > 0:
            report.type_warnings.append(f"{n_invalid} data(s) inválida(s) — serão removidas.")
        valid_dates = parsed.dropna()
        if len(valid_dates) > 0:
            report.date_range = (str(valid_dates.min().date()), str(valid_dates.max().date()))
        freq_info = detect_series_frequency(parsed)
        report.detected_freq_hours = freq_info["freq_hours"]
    except Exception as e:
        report.type_warnings.append(f"Erro ao processar datas: {e}")

    # Instrumento
    if "instrumento" in df.columns:
        report.n_instruments = int(df["instrumento"].nunique())
        counts = df["instrumento"].value_counts()
        sparse = counts[counts < 30].index.tolist()
        if sparse:
            report.info_messages.append(
                f"Instrumentos com < 30 leituras (modelo menos estável): {sparse}"
            )

    # Leitura numérica
    if "leitura" in df.columns:
        non_num = int(pd.to_numeric(df["leitura"], errors="coerce").isna().sum())
        if non_num > 0:
            report.type_warnings.append(f"Coluna 'leitura': {non_num} valores não numéricos.")

    return report


def validate_business_requirements(df: pd.DataFrame) -> list:
    """
    Verifica as colunas de negócio obrigatórias após montagem da base completa.

    Regras:
    - pluviometria: OBRIGATÓRIA — bloqueia se ausente ou toda nula
    - tipo_material: OBRIGATÓRIO — bloqueia se ausente ou toda nula
    - permeabilidade: OBRIGATÓRIA — bloqueia se ausente ou toda nula
    - nivel_reservatorio: OPCIONAL — apenas avisa se ausente

    Returns:
        Lista de warnings (não bloqueantes).

    Raises:
        ValueError: com mensagem clara orientando o usuário.
    """
    errors   = []
    warnings = []

    for col in ["pluviometria", "tipo_material", "permeabilidade"]:
        if col not in df.columns:
            errors.append(
                f"'{col}' não encontrada. Forneça este dado nas etapas de entrada."
            )
        elif df[col].isna().all():
            errors.append(f"'{col}' está completamente vazia.")

    if "nivel_reservatorio" not in df.columns or df["nivel_reservatorio"].isna().all():
        warnings.append(
            "'nivel_reservatorio' não informado — será omitido como feature. "
            "Recomendado para barragens com reservatório relevante."
        )

    if errors:
        raise ValueError(
            "Não é possível executar o modelo. Corrija os seguintes problemas:\n\n"
            + "\n".join(f"  ❌ {e}" for e in errors)
        )

    return warnings


def check_external_data_coverage(
    df_main: pd.DataFrame,
    df_ext: pd.DataFrame,
    date_col: str = "data",
    ext_cols: list = None,
) -> list:
    """Verifica cobertura temporal entre base principal e dados externos."""
    msgs = []
    try:
        main_dates = pd.to_datetime(df_main[date_col], errors="coerce").dropna()
        ext_dates  = pd.to_datetime(df_ext[date_col],  errors="coerce").dropna()
        if ext_dates.min() > main_dates.min():
            msgs.append(
                f"Dados externos começam em {ext_dates.min().date()}, "
                f"mas a série de leituras começa em {main_dates.min().date()}. "
                "Período inicial sem dado externo."
            )
        if ext_dates.max() < main_dates.max():
            msgs.append(
                f"Dados externos terminam em {ext_dates.max().date()}, "
                f"mas a série de leituras vai até {main_dates.max().date()}. "
                "Período final sem dado externo."
            )
        if ext_cols:
            for col in ext_cols:
                if col in df_ext.columns:
                    pct = 100 * float(df_ext[col].isna().sum()) / max(len(df_ext), 1)
                    if pct > 5:
                        msgs.append(
                            f"Coluna '{col}' nos dados externos: {pct:.1f}% de valores nulos."
                        )
    except Exception as e:
        msgs.append(f"Erro ao verificar cobertura: {e}")
    return msgs
