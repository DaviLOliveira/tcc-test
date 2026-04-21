"""
permeability_db.py — PiezoPrev Final
Módulo de leitura e consulta da base externa de permeabilidade.

A base de referência NÃO está hardcoded no código Python.
Ela é lida de um arquivo externo: permeabilidade_ref.csv (ou .xlsx)
localizado na mesma pasta do projeto.

Estrutura esperada do arquivo externo:
    tipo_material  | k_cms   | k_min  | k_max  | fonte        | observacao
    Argila         | 1.0e-07 | 1.0e-9 | 1.0e-6 | Das (2010)   | Solo coesivo...

Fluxo:
1. O sistema tenta carregar permeabilidade_ref.csv na inicialização.
2. Se não encontrar o CSV, tenta permeabilidade_ref.xlsx.
3. Se não encontrar nenhum, retorna DataFrame vazio e exibe aviso.
4. A busca por material é tolerante a variações de grafia (sem acentos, lowercase).
5. Qualquer alteração no arquivo externo é refletida após restart da aplicação.
"""

import io
import logging
import unicodedata
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# Nome padrão do arquivo de referência (mesma pasta do projeto)
_REF_CSV  = Path(__file__).parent / "permeabilidade_ref.csv"
_REF_XLSX = Path(__file__).parent / "permeabilidade_ref.xlsx"

# Colunas obrigatórias no arquivo externo
_REQUIRED_COLS = ["tipo_material", "k_cms"]
# Colunas opcionais
_OPTIONAL_COLS = ["k_min", "k_max", "fonte", "observacao"]

# Cache em memória (recarregado a cada chamada de load_permeability_db se forçado)
_DB_CACHE: Optional[pd.DataFrame] = None


# ---------------------------------------------------------------------------
# Normalização de texto para busca tolerante
# ---------------------------------------------------------------------------

def _normalize(s: str) -> str:
    """Remove acentos, converte para lowercase e elimina espaços duplos."""
    s = str(s).lower().strip()
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return " ".join(s.split())


# ---------------------------------------------------------------------------
# Carregamento do arquivo externo
# ---------------------------------------------------------------------------

def load_permeability_db(
    filepath: Optional[str] = None,
    force_reload: bool = False,
) -> pd.DataFrame:
    """
    Carrega a base de referência de permeabilidade de arquivo externo.

    Tenta carregar nesta ordem:
    1. filepath fornecido explicitamente (se informado)
    2. permeabilidade_ref.csv na pasta do projeto
    3. permeabilidade_ref.xlsx na pasta do projeto

    Args:
        filepath: Caminho explícito para o arquivo (opcional).
                  Pode ser string de caminho ou bytes (upload via Streamlit).
        force_reload: Se True, ignora cache e recarrega do disco.

    Returns:
        DataFrame com a base de permeabilidade.
        DataFrame vazio se o arquivo não for encontrado.
    """
    global _DB_CACHE

    if _DB_CACHE is not None and not force_reload and filepath is None:
        return _DB_CACHE

    df = pd.DataFrame()

    # 1. Arquivo explicitamente fornecido
    if filepath is not None:
        try:
            if isinstance(filepath, (str, Path)):
                path = Path(filepath)
                if path.suffix.lower() == ".csv":
                    df = pd.read_csv(str(path))
                else:
                    df = pd.read_excel(str(path))
                logger.info(f"Base de permeabilidade carregada de: {path}")
            elif isinstance(filepath, (bytes, io.BytesIO)):
                raw = io.BytesIO(filepath) if isinstance(filepath, bytes) else filepath
                # Tenta CSV primeiro, depois Excel
                try:
                    df = pd.read_csv(raw)
                except Exception:
                    raw.seek(0)
                    df = pd.read_excel(raw)
                logger.info("Base de permeabilidade carregada via upload.")
        except Exception as e:
            logger.error(f"Erro ao carregar base de permeabilidade de '{filepath}': {e}")
            df = pd.DataFrame()

    # 2. Arquivo padrão CSV
    elif _REF_CSV.exists():
        try:
            df = pd.read_csv(str(_REF_CSV))
            logger.info(f"Base de permeabilidade carregada: {_REF_CSV} ({len(df)} registros)")
        except Exception as e:
            logger.error(f"Erro ao ler {_REF_CSV}: {e}")

    # 3. Arquivo padrão XLSX
    elif _REF_XLSX.exists():
        try:
            df = pd.read_excel(str(_REF_XLSX))
            logger.info(f"Base de permeabilidade carregada: {_REF_XLSX} ({len(df)} registros)")
        except Exception as e:
            logger.error(f"Erro ao ler {_REF_XLSX}: {e}")

    else:
        logger.warning(
            "Arquivo de referência de permeabilidade não encontrado. "
            f"Esperado: {_REF_CSV} ou {_REF_XLSX}. "
            "Crie o arquivo ou faça upload na interface. "
            "O usuário precisará informar a permeabilidade manualmente."
        )

    # Validar e limpar
    if not df.empty:
        df = _validate_and_clean(df)

    _DB_CACHE = df
    return df


def _validate_and_clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Valida e limpa o DataFrame carregado da base externa.
    - Normaliza nomes de colunas
    - Verifica colunas obrigatórias
    - Converte tipos numéricos
    - Adiciona colunas opcionais ausentes como vazio
    """
    # Normalizar nomes de colunas
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    missing = [c for c in _REQUIRED_COLS if c not in df.columns]
    if missing:
        logger.error(
            f"Base de permeabilidade inválida — colunas obrigatórias ausentes: {missing}. "
            f"Colunas encontradas: {list(df.columns)}"
        )
        return pd.DataFrame()

    # Converter numéricos
    for col in ["k_cms", "k_min", "k_max"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Adicionar opcionais ausentes
    for col in _OPTIONAL_COLS:
        if col not in df.columns:
            df[col] = ""

    # Remover linhas sem k_cms válido
    before = len(df)
    df = df.dropna(subset=["k_cms"])
    if len(df) < before:
        logger.warning(f"Base de permeabilidade: {before - len(df)} linhas removidas por k_cms inválido.")

    # Adicionar coluna normalizada para busca
    df["_tipo_norm"] = df["tipo_material"].apply(_normalize)

    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Consulta por tipo de material
# ---------------------------------------------------------------------------

def lookup_permeability(
    soil_name: str,
    db: Optional[pd.DataFrame] = None,
) -> Optional[dict]:
    """
    Busca permeabilidade de referência pelo nome do solo/material.

    A busca é tolerante a variações de grafia, acentos e capitalização.
    Estratégia de correspondência (em ordem de prioridade):
    1. Correspondência exata (normalizada)
    2. O nome buscado está contido no nome do banco
    3. O nome do banco está contido no nome buscado

    Args:
        soil_name: Nome do solo informado pelo usuário.
        db: DataFrame da base de permeabilidade. Se None, carrega automaticamente.

    Returns:
        Dicionário com os dados do material encontrado, ou None se não encontrado.
        Campos: tipo_material, k_cms, k_min, k_max, fonte, observacao
    """
    if not soil_name or not str(soil_name).strip():
        return None

    if db is None:
        db = load_permeability_db()

    if db.empty:
        return None

    key = _normalize(soil_name)

    # 1. Correspondência exata
    exact = db[db["_tipo_norm"] == key]
    if not exact.empty:
        return _row_to_dict(exact.iloc[0])

    # 2. Nome buscado está contido no banco
    contained_in = db[db["_tipo_norm"].str.contains(key, regex=False, na=False)]
    if not contained_in.empty:
        return _row_to_dict(contained_in.iloc[0])

    # 3. Nome do banco está contido no buscado
    reverse = db[db["_tipo_norm"].apply(lambda x: x in key)]
    if not reverse.empty:
        # Pega o mais específico (mais longo)
        best = reverse.loc[reverse["_tipo_norm"].str.len().idxmax()]
        return _row_to_dict(best)

    return None


def _row_to_dict(row: pd.Series) -> dict:
    """Converte uma linha do DataFrame em dicionário de resultado."""
    return {
        "tipo_material": str(row.get("tipo_material", "")),
        "k_cms":         float(row.get("k_cms", 0)),
        "k_min":         float(row["k_min"]) if pd.notna(row.get("k_min")) else None,
        "k_max":         float(row["k_max"]) if pd.notna(row.get("k_max")) else None,
        "fonte":         str(row.get("fonte", "")).strip(),
        "observacao":    str(row.get("observacao", "")).strip(),
    }


# ---------------------------------------------------------------------------
# Lista de solos disponíveis
# ---------------------------------------------------------------------------

def get_soil_options(db: Optional[pd.DataFrame] = None) -> list:
    """
    Retorna lista de solos disponíveis na base externa, ordenada alfabeticamente.

    Args:
        db: DataFrame da base. Se None, carrega automaticamente.

    Returns:
        Lista de strings com nomes dos materiais.
    """
    if db is None:
        db = load_permeability_db()
    if db.empty:
        return []
    return sorted(db["tipo_material"].dropna().unique().tolist())


# ---------------------------------------------------------------------------
# Status da base de permeabilidade
# ---------------------------------------------------------------------------

def get_db_status() -> dict:
    """
    Retorna informações sobre o estado da base de permeabilidade carregada.

    Returns:
        Dict com: loaded, n_records, filepath, columns, warnings
    """
    db = load_permeability_db()
    status = {
        "loaded": not db.empty,
        "n_records": len(db),
        "csv_exists": _REF_CSV.exists(),
        "xlsx_exists": _REF_XLSX.exists(),
        "expected_path_csv": str(_REF_CSV),
        "expected_path_xlsx": str(_REF_XLSX),
        "columns": list(db.columns) if not db.empty else [],
        "materials": get_soil_options(db) if not db.empty else [],
    }
    return status
