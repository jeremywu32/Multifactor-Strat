"""WRDS CRSP CIZ S&P 500 downloader.

The original user-supplied wrds.py has been adapted to retrieve daily factor inputs.
No remote queries are made on import. Your WRDS entitlement is required to download.

Install: pip install pandas sqlalchemy psycopg2-binary python-dotenv pyarrow
Put WRDS_USER and WRDS_PASSWORD into .env next to this script.
"""

import hashlib
import os
from functools import lru_cache
from pathlib import Path

import pandas as pd
from sqlalchemy import URL, bindparam, create_engine as sqlalchemy_create_engine, text
from dotenv import load_dotenv

THIS_DIR = Path(__file__).resolve().parent
ENV_PATH = THIS_DIR / '.env'
CACHE_DIR = THIS_DIR / 'cache'

PERMNO = 'permno'
DATE = 'dlycaldt'

SP500_HISTORY_SQL = """
    SELECT permno, mbrstartdt, mbrenddt
    FROM crsp_a_indexes.dsp500list_v2
    ORDER BY permno, mbrstartdt
"""

# CRSP CIZ (v2). dlyprcvol is a dollar-volume PROXY (price times shares traded),
# NOT actual trade-by-trade dollar volume / VWAP data.
DAILY_STOCK_DATA_SQL = """
    SELECT
        permno, permco, ticker, siccd, dlycaldt,
        dlyopen, dlyhigh, dlylow, dlyclose, dlyprc,
        dlyret, dlyretx, dlyvol, dlyprcvol,
        shrout, dlycap, dlycumfacpr, dlycumfacshr,
        dlydelflg
    FROM crsp.dsf_v2
    WHERE permno IN :permnos
      AND dlycaldt BETWEEN :start AND :end
    ORDER BY permno, dlycaldt
"""


@lru_cache(maxsize=1)
def create_engine():
    load_dotenv(ENV_PATH)
    user, password = os.getenv('WRDS_USER'), os.getenv('WRDS_PASSWORD')
    if not user or not password:
        raise RuntimeError(
            f'Missing WRDS_USER / WRDS_PASSWORD. Create {ENV_PATH} first.'
        )
    url = URL.create(
        drivername='postgresql+psycopg2',
        username=user,
        password=password,
        host='wrds-pgdata.wharton.upenn.edu',
        port=9737,
        database='wrds',
    )
    return sqlalchemy_create_engine(url, pool_pre_ping=True)


def cached_query(name, sql, date_columns_to_parse, params_dict=None,
                 force_refresh=False, expand_permnos=False):
    # Including SQL in the cache key avoids accidentally loading an old 3-column
    # parquet after changing the SELECT list.
    query_hash = hashlib.sha256(sql.encode('utf-8')).hexdigest()[:10]
    cache_path = CACHE_DIR / f'{name}_{query_hash}.parquet'
    if cache_path.exists() and not force_refresh:
        print(f'Loading from cache: {cache_path}')
        return pd.read_parquet(cache_path)

    statement = text(sql)
    if expand_permnos:
        statement = statement.bindparams(bindparam('permnos', expanding=True))

    print(f'Downloading {name} from WRDS...')
    with create_engine().connect() as connection:
        df = pd.read_sql(
            statement,
            connection,
            params=params_dict,
            parse_dates=date_columns_to_parse,
        )
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path, index=False)
    print(f'Saved {len(df):,} rows to {cache_path}')
    return df


def load_sp500_history(force_refresh=False):
    return cached_query(
        'sp500_historical_constituents',
        SP500_HISTORY_SQL,
        ['mbrstartdt', 'mbrenddt'],
        force_refresh=force_refresh,
    )


def load_daily_stock_data(permnos, start, end, force_refresh=False):
    permnos = tuple(sorted({int(p) for p in permnos}))
    if not permnos:
        raise ValueError('permnos is empty: no matching S&P 500 constituents.')
    start, end = pd.Timestamp(start).date(), pd.Timestamp(end).date()
    if start > end:
        raise ValueError('start must be <= end')

    permno_hash = hashlib.sha256(
        ','.join(map(str, permnos)).encode('utf-8')
    ).hexdigest()[:12]
    return cached_query(
        f'sp500_daily_{start}_{end}_{permno_hash}',
        DAILY_STOCK_DATA_SQL,
        [DATE],
        params_dict={'permnos': permnos, 'start': start, 'end': end},
        force_refresh=force_refresh,
        expand_permnos=True,
    )


def eligible_permnos(membership, eval_start, eval_end):
    """All securities with any S&P 500 membership during the evaluation span."""
    start = pd.Timestamp(eval_start)
    end = pd.Timestamp(eval_end)
    mask = (membership['mbrstartdt'].le(end) &
            (membership['mbrenddt'].isna() | membership['mbrenddt'].ge(start)))
    return sorted(membership.loc[mask, 'permno'].dropna().astype(int).unique())


def download_years(eval_start='2015-01-01', eval_end='2025-12-31',
                   history_start='2010-01-01', output_dir='sp500_data',
                   force_refresh=False):
    """Download data by year, including a pre-evaluation warm-up period.

    Raw data is deliberately NOT cut to S&P membership dates, so historical
    features can use observations before a stock joined the index.
    """
    eval_start, eval_end = pd.Timestamp(eval_start), pd.Timestamp(eval_end)
    history_start = pd.Timestamp(history_start)
    if not (history_start <= eval_start <= eval_end):
        raise ValueError('Expected history_start <= eval_start <= eval_end')

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    membership = load_sp500_history(force_refresh=force_refresh)
    membership.to_parquet(output_dir / 'sp500_membership.parquet', index=False)
    permnos = eligible_permnos(membership, eval_start, eval_end)
    print(f'{len(permnos)} distinct PERMNOs in evaluation universe')

    outputs = []
    all_data = []

    for year in range(history_start.year, eval_end.year + 1):
        year_start = max(
            history_start,
            pd.Timestamp(year=year, month=1, day=1)
        )
        year_end = min(
            eval_end,
            pd.Timestamp(year=year, month=12, day=31)
        )

        if year_start > year_end:
            continue

        df = load_daily_stock_data(
            permnos,
            year_start,
            year_end,
            force_refresh=force_refresh
        )

        # Add annual data to a list
        all_data.append(df)

        print(
            f"{year}: {len(df):,} rows, "
            f"{df['permno'].nunique()} securities"
        )

    # Concatenate once, after the loop
    full_data = pd.concat(
        all_data,
        ignore_index=True
    )

    # Sort the full panel
    full_data = full_data.sort_values(
        ["permno", "dlycaldt"]
    ).reset_index(drop=True)

    # Export a single parquet file
    target = output_dir / "sp500_full_data.pq"

    full_data.to_parquet(
        target,
        engine="pyarrow",
        compression="zstd",
        index=False
    )

    print(f"Saved {len(full_data):,} rows")
    print(f"Number of securities: {full_data['permno'].nunique()}")
    print(f"Saved to: {target}")

    return full_data


if __name__ == '__main__':
    download_years()
