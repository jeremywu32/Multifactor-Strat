"""Configure these three dates, then run: python run_sp500_download.py"""
from wrds_sp500 import download_years

if __name__ == '__main__':
    full_data = download_years(
        eval_start='2015-01-01',     # factor evaluation starts
        eval_end='2025-12-31',       # factor evaluation ends
        history_start='2010-01-01',  # 5-year cost-basis / momentum warm-up
        output_dir='sp500_data',
        force_refresh=False,
    )
    
