"""
preprocessing.py
----------------
Pré-processamento da base consolidada.

Responsabilidades (após carregamento):
- Converter tipos
- Ordenar temporalmente
- Remover duplicatas
- Codificar categóricas (encoding salvo para uso em predição)
- NÃO interpolar (isso fica em quality.py)
- NÃO criar features (isso fica em features.py)

Leakage safety:
- LabelEncoder é ajustado SOMENTE no conjunto de treino e aplicado no teste.
  O pipeline de pré-processamento aqui apenas prepara o encoder para uso posterior.
  Na prática, codifica a base completa com todos os valores conhecidos, o que
  é seguro para categóricas estáticas (tipo_material, tipo_instrumento).
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from config import REQUIRED_COLUMNS, OPTIONAL_COLUMNS, CATEGORICAL_COLS

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Parsing de datas
# ---------------------------------------------------------------------------

def parse_dates(df: pd.DataFrame, date_col: str = "data") -> tuple[pd.DataFrame, int]:
    """
    Converte a coluna de data para datetime64.

    Args:
        df: DataFrame com coluna de data.
        date_col: Nome da coluna.

    Returns:
        Tuple (DataFrame com datas convertidas, número de registros inválidos removidos).
    """
    df = df.copy()
    original_len = len(df)

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce", dayfirst=True)
    n_invalid = df[date_col].isna().sum()

    if n_invalid > 0:
        logger.warning(
            f"Coluna '{date_col}': {n_invalid} registros com datas inválidas — serão removidos."
        )
        df = df.dropna(subset=[date_col])

    return df, n_invalid


# ---------------------------------------------------------------------------
# Remoção de duplicatas
# ---------------------------------------------------------------------------

def remove_duplicates(
    df: pd.DataFrame,
    subset: list[str] = None,
    keep: str = "last",
) -> tuple[pd.DataFrame, int]:
    """
    Remove registros duplicados por (instrumento, data).

    Mantém o último registro por padrão (mais recente / revisado).

    Args:
        df: DataFrame ordenado.
        subset: Colunas para identificar duplicata. Padrão: ['instrumento', 'data'].
        keep: 'first' ou 'last'.

    Returns:
        Tuple (DataFrame sem duplicatas, número de duplicatas removidas).
    """
    if subset is None:
        subset = ["instrumento", "data"]

    # Apenas usa colunas que existem
    subset = [c for c in subset if c in df.columns]
    before = len(df)
    df = df.drop_duplicates(subset=subset, keep=keep)
    removed = before - len(df)

    if removed > 0:
        logger.info(f"Duplicatas removidas: {removed} (chave: {subset})")

    return df, removed


# ---------------------------------------------------------------------------
# Ordenação temporal
# ---------------------------------------------------------------------------

def sort_temporally(
    df: pd.DataFrame,
    group_col: str = "instrumento",
    date_col: str = "data",
) -> pd.DataFrame:
    """
    Ordena por instrumento e data de forma crescente.
    Essencial antes de qualquer operação de lag ou rolling.
    """
    return df.sort_values([group_col, date_col]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Conversão de tipos
# ---------------------------------------------------------------------------

def enforce_numeric_types(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Converte colunas numéricas esperadas que estejam como object/string.

    Colunas numéricas esperadas: leitura, pluviometria, nivel_reservatorio,
    permeabilidade, profundidade, cota_instalacao.

    Valores não-numéricos são convertidos para NaN.

    Returns:
        Tuple (DataFrame com tipos corrigidos, lista de colunas corrigidas).
    """
    numeric_expected = [
        "leitura", "pluviometria", "nivel_reservatorio",
        "permeabilidade", "profundidade", "cota_instalacao",
    ]

    df = df.copy()
    corrected = []

    for col in numeric_expected:
        if col in df.columns and df[col].dtype == object:
            original_nulls = df[col].isna().sum()
            df[col] = pd.to_numeric(df[col], errors="coerce")
            new_nulls = df[col].isna().sum()
            if new_nulls > original_nulls:
                logger.warning(
                    f"Coluna '{col}': {new_nulls - original_nulls} valores "
                    "não numéricos convertidos para NaN."
                )
            corrected.append(col)

    return df, corrected


# ---------------------------------------------------------------------------
# Codificação de variáveis categóricas
# ---------------------------------------------------------------------------

def encode_categoricals(
    df: pd.DataFrame,
    cat_cols: Optional[list[str]] = None,
    existing_encoders: Optional[dict] = None,
) -> tuple[pd.DataFrame, dict]:
    """
    Codifica variáveis categóricas com LabelEncoder.

    Aceita encoders já ajustados (para aplicar nos mesmos labels usados no treino).
    Quando existing_encoders é None, ajusta novos encoders na base completa.

    Novas categorias não vistas no treino são tratadas como 'Desconhecido'
    para evitar erro de transformação.

    Args:
        df: DataFrame.
        cat_cols: Colunas categóricas. Se None, usa CATEGORICAL_COLS.
        existing_encoders: Encoders já ajustados (para predição / teste).

    Returns:
        Tuple (DataFrame com colunas _enc adicionadas, dicionário de encoders).
    """
    df = df.copy()
    encoders = existing_encoders.copy() if existing_encoders else {}

    if cat_cols is None:
        cat_cols = [c for c in CATEGORICAL_COLS if c in df.columns]

    for col in cat_cols:
        df[col] = df[col].astype(str).fillna("Desconhecido")

        if col in encoders:
            # Aplicar encoder existente — mapear unseen para "Desconhecido"
            le = encoders[col]
            known_labels = set(le.classes_)
            df[col] = df[col].apply(lambda x: x if x in known_labels else "Desconhecido")
            df[f"{col}_enc"] = le.transform(df[col])
        else:
            # Ajustar novo encoder
            le = LabelEncoder()
            # Garantir que "Desconhecido" esteja nas classes (para uso futuro)
            all_labels = sorted(set(df[col].tolist()) | {"Desconhecido"})
            le.fit(all_labels)
            df[f"{col}_enc"] = le.transform(df[col])
            encoders[col] = le
            logger.info(
                f"Encoder ajustado para '{col}': {len(le.classes_)} classes — {list(le.classes_)}"
            )

    return df, encoders


# ---------------------------------------------------------------------------
# Pipeline de pré-processamento completo
# ---------------------------------------------------------------------------

class PreprocessingResult:
    """Resultado estruturado do pré-processamento."""
    __slots__ = [
        "df", "encoders", "n_invalid_dates", "n_duplicates_removed",
        "corrected_types", "cat_cols_encoded",
    ]

    def __init__(self, df, encoders, n_invalid_dates=0, n_duplicates_removed=0,
                 corrected_types=None, cat_cols_encoded=None):
        self.df = df
        self.encoders = encoders
        self.n_invalid_dates = n_invalid_dates
        self.n_duplicates_removed = n_duplicates_removed
        self.corrected_types = corrected_types or []
        self.cat_cols_encoded = cat_cols_encoded or []


def run_preprocessing(
    df: pd.DataFrame,
    date_col: str = "data",
    group_col: str = "instrumento",
    encode_cats: bool = True,
    existing_encoders: Optional[dict] = None,
) -> PreprocessingResult:
    """
    Pipeline de pré-processamento.

    Etapas:
    1. Conversão de tipos numéricos
    2. Parsing de datas
    3. Remoção de duplicatas
    4. Ordenação temporal
    5. Codificação de categóricas (quando encode_cats=True)

    Args:
        df: DataFrame bruto.
        date_col: Coluna de data.
        group_col: Coluna de instrumento.
        encode_cats: Se True, codifica variáveis categóricas.
        existing_encoders: Encoders ajustados anteriormente (para predição).

    Returns:
        PreprocessingResult com df processado e metadados.
    """
    logger.info(f"=== Pré-processamento iniciado: {len(df):,} registros ===")

    df, corrected_types = enforce_numeric_types(df)
    df, n_invalid = parse_dates(df, date_col)
    df, n_dupes = remove_duplicates(df)
    df = sort_temporally(df, group_col, date_col)

    encoders = {}
    cat_cols_encoded = []
    if encode_cats:
        df, encoders = encode_categoricals(df, existing_encoders=existing_encoders)
        cat_cols_encoded = [c for c in df.columns if c.endswith("_enc")]

    logger.info(
        f"=== Pré-processamento concluído: {len(df):,} registros | "
        f"Datas inválidas: {n_invalid} | Duplicatas: {n_dupes} | "
        f"Colunas codificadas: {cat_cols_encoded} ==="
    )

    return PreprocessingResult(
        df=df,
        encoders=encoders,
        n_invalid_dates=n_invalid,
        n_duplicates_removed=n_dupes,
        corrected_types=corrected_types,
        cat_cols_encoded=cat_cols_encoded,
    )
