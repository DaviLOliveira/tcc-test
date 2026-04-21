"""
data_loader.py — PiezoPrev Final (corrigido)

Correções aplicadas nesta versão:
─────────────────────────────────────────────────────────────────────────────
1. _read_file()
   - Detecta separador com Sniffer + fallback por contagem
   - Detecta codificação automaticamente
   - Usa engine="c" (padrão pandas) — compatível com low_memory=False
   - Remove completamente engine="python" que causava erro com low_memory

2. merge_external_data()
   - Converte colunas externas para numérico antes da agregação (evita erro
     "DataError: No numeric types to aggregate" quando colunas são object)
   - Normaliza datas para date (sem componente de hora) antes do merge,
     evitando que diferenças de hora impeçam o join
   - Log detalhado do número de linhas antes e depois do merge

3. load_main_data() / load_external_data()
   - Aplica coerção numérica nas colunas esperadas logo após o carregamento
   - Informa colunas encontradas após mapeamento de aliases
─────────────────────────────────────────────────────────────────────────────
"""

import csv
import io
import logging
import unicodedata
import pandas as pd
import numpy as np

from config import COLUMN_ALIASES

logger = logging.getLogger(__name__)

# Colunas que devem ser numéricas após o carregamento
_NUMERIC_COLS_EXPECTED = [
    "leitura", "pluviometria", "nivel_reservatorio",
    "permeabilidade", "profundidade", "cota_instalacao",
]


# ---------------------------------------------------------------------------
# Normalização de nomes de colunas
# ---------------------------------------------------------------------------

def _normalize(s: str) -> str:
    s = str(s).lower().strip()
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return " ".join(s.split())


# ---------------------------------------------------------------------------
# Detecção de separador CSV
# ---------------------------------------------------------------------------

def _detect_csv_separator(raw_bytes: bytes) -> str:
    """
    Detecta o separador de um CSV a partir dos primeiros bytes.
    Estratégia: Sniffer → fallback por contagem de colunas.
    """
    sample_str = ""
    for encoding in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
        try:
            sample_str = raw_bytes[:4096].decode(encoding)
            break
        except (UnicodeDecodeError, Exception):
            continue

    if not sample_str:
        return ","

    # 1. Sniffer
    try:
        dialect = csv.Sniffer().sniff(sample_str, delimiters=",;\t|")
        if dialect.delimiter in (",", ";", "\t", "|"):
            return dialect.delimiter
    except csv.Error:
        pass

    # 2. Fallback por contagem
    first_line = sample_str.split("\n")[0]
    candidates = {
        ",":  len(first_line.split(",")),
        ";":  len(first_line.split(";")),
        "\t": len(first_line.split("\t")),
        "|":  len(first_line.split("|")),
    }
    best = max(candidates, key=candidates.get)
    return best if candidates[best] > 1 else ","


def _detect_encoding(raw_bytes: bytes) -> str:
    for enc in ("utf-8-sig", "utf-8", "latin-1", "cp1252", "iso-8859-1"):
        try:
            raw_bytes[:2048].decode(enc)
            return enc
        except (UnicodeDecodeError, Exception):
            continue
    return "utf-8"


# ---------------------------------------------------------------------------
# Coerção de colunas numéricas
# ---------------------------------------------------------------------------

def _coerce_numeric_columns(df: pd.DataFrame, cols: list = None) -> pd.DataFrame:
    """
    Converte para numérico as colunas esperadas que estejam como object.
    Valores não convertíveis viram NaN (sem erro).

    Args:
        df: DataFrame.
        cols: Colunas a converter. Se None, usa _NUMERIC_COLS_EXPECTED.

    Returns:
        DataFrame com colunas convertidas.
    """
    if cols is None:
        cols = _NUMERIC_COLS_EXPECTED
    df = df.copy()
    for col in cols:
        if col in df.columns and df[col].dtype == object:
            original_nulls = int(df[col].isna().sum())
            df[col] = pd.to_numeric(df[col], errors="coerce")
            new_nulls = int(df[col].isna().sum())
            if new_nulls > original_nulls:
                logger.warning(
                    f"Coluna '{col}': {new_nulls - original_nulls} valores "
                    "não numéricos convertidos para NaN."
                )
    return df


# ---------------------------------------------------------------------------
# Leitura de arquivo (CSV ou Excel) — sem engine="python"
# ---------------------------------------------------------------------------

def _read_file(source, filename: str) -> pd.DataFrame:
    """
    Lê CSV ou Excel de caminho em disco ou de bytes (upload Streamlit).

    Para CSV:
    - Detecta separador e codificação automaticamente
    - Usa engine padrão ("c") — compatível com low_memory=False
    - NÃO usa engine="python" com low_memory (combinação inválida no pandas)

    Para Excel:
    - pd.read_excel() diretamente.
    """
    fn = str(filename).lower().strip()

    # ── Leitura de disco ───────────────────────────────────────────────────
    if isinstance(source, str) or hasattr(source, "__fspath__"):
        path = str(source)
        if fn.endswith(".csv"):
            with open(path, "rb") as f:
                raw = f.read()
            sep      = _detect_csv_separator(raw)
            encoding = _detect_encoding(raw)
            df = pd.read_csv(path, sep=sep, encoding=encoding, low_memory=False)
            logger.info(f"CSV (disco): sep={repr(sep)}, enc={encoding}, shape={df.shape}")
            return df
        elif fn.endswith((".xlsx", ".xls")):
            df = pd.read_excel(path)
            logger.info(f"Excel (disco): shape={df.shape}")
            return df
        else:
            raise ValueError(
                f"Formato não suportado: '{filename}'. Use .csv, .xlsx ou .xls."
            )

    # ── Leitura de bytes (upload Streamlit) ────────────────────────────────
    raw_bytes = source if isinstance(source, bytes) else source.read()

    if fn.endswith(".csv"):
        sep      = _detect_csv_separator(raw_bytes)
        encoding = _detect_encoding(raw_bytes)
        df = pd.read_csv(
            io.BytesIO(raw_bytes),
            sep=sep,
            encoding=encoding,
            low_memory=False,   # engine padrão "c" — sem conflito
        )
        logger.info(f"CSV (bytes): sep={repr(sep)}, enc={encoding}, shape={df.shape}")
        return df
    elif fn.endswith((".xlsx", ".xls")):
        df = pd.read_excel(io.BytesIO(raw_bytes))
        logger.info(f"Excel (bytes): shape={df.shape}")
        return df
    else:
        raise ValueError(
            f"Formato não suportado: '{filename}'. Use .csv, .xlsx ou .xls."
        )


# ---------------------------------------------------------------------------
# Mapeamento de nomes amigáveis de colunas
# ---------------------------------------------------------------------------

def apply_column_aliases(df: pd.DataFrame) -> tuple:
    rename_map = {}
    for col in df.columns:
        normalized = _normalize(col)
        if normalized in COLUMN_ALIASES:
            target = COLUMN_ALIASES[normalized]
            if target != col:
                rename_map[col] = target
    if rename_map:
        df = df.rename(columns=rename_map)
        for orig, mapped in rename_map.items():
            logger.info(f"Coluna renomeada: '{orig}' → '{mapped}'")
    return df, rename_map


# ---------------------------------------------------------------------------
# Funções públicas de carregamento
# ---------------------------------------------------------------------------

def load_main_data(source, filename: str) -> pd.DataFrame:
    """
    Carrega a base principal de leituras.
    Aplica mapeamento de aliases e coerção numérica.
    """
    df = _read_file(source, filename)
    df.columns = [str(c).strip() for c in df.columns]
    df, aliases = apply_column_aliases(df)
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    df = _coerce_numeric_columns(df)

    logger.info(
        f"Base principal: {df.shape[0]:,} linhas × {df.shape[1]} colunas | "
        f"Colunas: {list(df.columns)}"
    )
    return df


def load_external_data(source, filename: str) -> pd.DataFrame:
    """
    Carrega tabela de dados externos (pluviometria, reservatório etc.).
    Exige coluna 'data'. Aplica coerção numérica nas colunas esperadas.
    """
    df = _read_file(source, filename)
    df.columns = [str(c).strip() for c in df.columns]
    df, _ = apply_column_aliases(df)
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    df = _coerce_numeric_columns(df)

    if "data" not in df.columns:
        raise ValueError(
            "A tabela de dados externos precisa ter uma coluna de data. "
            f"Colunas encontradas: {list(df.columns)}. "
            "Nomes aceitos: 'data', 'date', 'Data da Leitura', 'timestamp'."
        )

    logger.info(f"Dados externos: {df.shape[0]:,} linhas | Colunas: {list(df.columns)}")
    return df


# ---------------------------------------------------------------------------
# Merge de dados externos — com normalização de datas e coerção numérica
# ---------------------------------------------------------------------------

def merge_external_data(
    df_main: pd.DataFrame,
    df_ext: pd.DataFrame,
    date_col: str = "data",
    how: str = "left",
    ext_cols: list = None,
) -> tuple:
    """
    Merge temporal entre base principal e dados externos.

    Melhorias desta versão:
    1. Normaliza ambas as datas para DATE (sem hora) antes do merge —
       evita que '2023-01-01 00:00:00' ≠ '2023-01-01' impeça o join.
    2. Converte colunas externas para numérico ANTES da agregação —
       evita "DataError: No numeric types to aggregate".
    3. Log com contagem de linhas antes e depois do merge.

    Args:
        df_main: Base principal.
        df_ext: Tabela externa com coluna 'data'.
        date_col: Nome da coluna de data.
        how: Tipo de merge ('left' = preserva todos os registros principais).
        ext_cols: Colunas externas a incluir. Se None, inclui todas exceto 'data'.

    Returns:
        Tuple (DataFrame merged, lista de colunas adicionadas).
    """
    if ext_cols is None:
        ext_cols = [c for c in df_ext.columns if c != date_col]

    new_cols = [c for c in ext_cols if c not in df_main.columns]
    if not new_cols:
        logger.info("Merge externo: nenhuma coluna nova para adicionar.")
        return df_main, []

    df_ext_sub = df_ext[[date_col] + new_cols].copy()

    # ── Coerção numérica antes de qualquer agregação ────────────────────────
    for col in new_cols:
        if col in df_ext_sub.columns and df_ext_sub[col].dtype == object:
            df_ext_sub[col] = pd.to_numeric(df_ext_sub[col], errors="coerce")
            logger.debug(f"Coluna externa '{col}' convertida para numérico.")

    # ── Normalizar datas: parsear com dayfirst=True e truncar para DATE ────
    # Isso garante que '01/01/2023' == '2023-01-01 08:00:00' no merge
    df_main = df_main.copy()
    df_main[date_col] = pd.to_datetime(
        df_main[date_col], errors="coerce", dayfirst=True
    ).dt.normalize()   # trunca para meia-noite (00:00:00)

    df_ext_sub[date_col] = pd.to_datetime(
        df_ext_sub[date_col], errors="coerce", dayfirst=True
    ).dt.normalize()

    # ── Remover datas inválidas na tabela externa ──────────────────────────
    n_invalid_ext = int(df_ext_sub[date_col].isna().sum())
    if n_invalid_ext > 0:
        logger.warning(
            f"Dados externos: {n_invalid_ext} datas inválidas removidas "
            "antes do merge."
        )
        df_ext_sub = df_ext_sub.dropna(subset=[date_col])

    # ── Deduplicar por data (média das colunas numéricas) ──────────────────
    if df_ext_sub[date_col].duplicated().any():
        n_dupes = int(df_ext_sub[date_col].duplicated().sum())
        logger.warning(
            f"Dados externos: {n_dupes} datas duplicadas — "
            "calculando média por data."
        )
        # Seleciona apenas colunas numéricas para a agregação
        numeric_ext_cols = [
            c for c in new_cols
            if c in df_ext_sub.columns
            and pd.api.types.is_numeric_dtype(df_ext_sub[c])
        ]
        if numeric_ext_cols:
            df_ext_sub = df_ext_sub.groupby(date_col, as_index=False)[numeric_ext_cols].mean()
        else:
            # Sem colunas numéricas: apenas remove duplicatas mantendo primeira
            logger.warning(
                "Nenhuma coluna numérica nos dados externos — "
                "mantendo primeiro valor por data."
            )
            df_ext_sub = df_ext_sub.drop_duplicates(subset=[date_col], keep="first")

    # ── Merge ──────────────────────────────────────────────────────────────
    n_before = len(df_main)
    df_merged = pd.merge(df_main, df_ext_sub, on=date_col, how=how)
    n_after   = len(df_merged)

    logger.info(
        f"Merge externo: {new_cols} adicionadas | "
        f"Linhas antes={n_before} | Linhas depois={n_after}"
    )

    # Alertas de nulos pós-merge
    for col in new_cols:
        if col in df_merged.columns:
            n_null = int(df_merged[col].isna().sum())
            if n_null > 0:
                pct = 100 * n_null / max(n_after, 1)
                msg = (
                    f"Coluna '{col}' após merge: {n_null} nulos ({pct:.1f}%). "
                )
                if pct > 50:
                    msg += (
                        "MAIS DE 50% NULOS — verificar alinhamento de datas "
                        "entre base principal e dados externos."
                    )
                    logger.warning(msg)
                else:
                    logger.info(msg)

    return df_merged, new_cols


# ---------------------------------------------------------------------------
# Merge de metadados de instrumentos
# ---------------------------------------------------------------------------

def merge_instrument_metadata(
    df_main: pd.DataFrame,
    df_meta: pd.DataFrame,
    instrument_col: str = "instrumento",
    meta_cols: list = None,
) -> tuple:
    if instrument_col not in df_meta.columns:
        raise ValueError(
            f"Tabela de metadados precisa ter coluna '{instrument_col}'. "
            f"Colunas encontradas: {list(df_meta.columns)}"
        )
    if meta_cols is None:
        meta_cols = [c for c in df_meta.columns if c != instrument_col]
    new_cols = [c for c in meta_cols if c not in df_main.columns]
    if not new_cols:
        return df_main, []
    df_meta_sub = (
        df_meta[[instrument_col] + new_cols]
        .drop_duplicates(subset=[instrument_col], keep="last")
    )
    df_merged = pd.merge(df_main, df_meta_sub, on=instrument_col, how="left")
    logger.info(f"Metadados adicionados: {new_cols}")
    return df_merged, new_cols


# ---------------------------------------------------------------------------
# Injeção de metadados via interface
# ---------------------------------------------------------------------------

def inject_instrument_constants(
    df: pd.DataFrame,
    tipo_instrumento: str,
    tipo_material: str,
    permeabilidade_cms: float,
    instrument_col: str = "instrumento",
) -> pd.DataFrame:
    """
    Injeta metadados do instrumento diretamente no DataFrame.
    Permeabilidade convertida de cm/s para m/s (padrão SI interno).
    """
    df = df.copy()
    df["tipo_instrumento"] = tipo_instrumento
    df["tipo_material"]    = tipo_material
    df["permeabilidade"]   = float(permeabilidade_cms) * 1e-2   # cm/s → m/s
    logger.info(
        f"Metadados injetados: tipo='{tipo_instrumento}', "
        f"material='{tipo_material}', k={permeabilidade_cms:.2e} cm/s"
    )
    return df
