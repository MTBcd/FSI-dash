# data_fetching.py
import pandas as pd
from ib_insync import IB, util, Stock
from fredapi import Fred
import logging
from selenium import webdriver
from selenium.webdriver.common.by import By
import time
import logging
import configparser


def get_ibkr_series(config):
    """Fetch data from IBKR."""
    try:
        ib_port = int(config['data']['ib_port'])
        ib_client_id = int(config['data']['ib_client_id'])
        start_date = config['data']['start_date']

        ib = IB()
        ib.connect('127.0.0.1', ib_port, clientId=ib_client_id)

        hyg = Stock('HYG', 'SMART', 'USD')
        lqd = Stock('LQD', 'SMART', 'USD')

        hyg_series = fetch_ibkr_data(ib, hyg, start_date)
        lqd_series = fetch_ibkr_data(ib, lqd, start_date)

        ib.disconnect()

        credit_spread = (hyg_series - lqd_series) / lqd_series

        return {
            'Credit Spread (HYG - LQD)': credit_spread
        }
    except Exception as e:
        logging.error(f"Error fetching IBKR data: {e}", exc_info=True)
        return {}

def fetch_ibkr_data(ib, contract, start_date, bar_size='1 day'):
    """Fetch historical data from IBKR for a given contract."""
    try:
        bars = ib.reqHistoricalData(
            contract,
            endDateTime='',
            durationStr='10 Y',
            barSizeSetting=bar_size,
            whatToShow='TRADES',
            useRTH=True,
            formatDate=1
        )
        df = util.df(bars)
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        return df[df.index >= pd.to_datetime(start_date)]['close']
    except Exception as e:
        logging.error(f"Error fetching historical data for {contract}: {e}", exc_info=True)
        return pd.Series()

def get_fred_series(config):
    """Fetch data from FRED."""
    try:
        fred_api_key = config['data']['fred_api_key']
        start_date = config['data']['start_date']

        fred = Fred(api_key=fred_api_key)

        def series(series_id):
            s = fred.get_series(series_id)
            s.index = pd.to_datetime(s.index)
            return s[s.index >= pd.to_datetime(start_date)]

        return {
            'VXV': series('VXVCLS'), # SPX 3M implied vol (CBOE VXV)
            'VIX': series('VIXCLS'),
            'USD Overnight Rate': series('OBFR'),
            '3M T-Bill': series('DTB3'),
            '10Y Yield': series('DGS10'),
            '2Y Yield': series('DGS2'),
            'USD Index': series('DTWEXBGS'),
            'FRED RRP': series('RRPONTSYD'),
            'US Corp OAS': series('BAMLC0A0CM'),
            'US HY OAS': series('BAMLH0A0HYM2'),
            'OVX': series('OVXCLS'),  # Crude Oil Volatility Index
            'GVZ': series('GVZCLS'),  # Gold Volatility Index
            '1Y Yield': series('DGS1')  # 1-Year Treasury Constant Maturity Rate
        }
    except Exception as e:
        logging.error(f"Error fetching FRED data: {e}", exc_info=True)
        return {}

def load_extended_csv_data(config):

    base_path = config['data']['csv_base_path']

    # Debugging: Check the base path
    print(f"Base path: {base_path}")

    try:
        tbill = pd.read_csv(f"{base_path}\\us_3m_tbill_yield.csv", index_col='Date', parse_dates=True)
        move = pd.read_csv(f"{base_path}\\move_index.csv", index_col='Date', parse_dates=True)
        overnight = pd.read_csv(f"{base_path}\\usd_overnight_rate.csv", index_col='Date', parse_dates=True)
        irs_1y = pd.read_csv(f"{base_path}\\usd_1y_irs.csv", index_col='Date', parse_dates=True)
        sofr_fut = pd.read_csv(f"{base_path}\\sofr_3m_futures.csv", index_col='Date', parse_dates=True)

        return {
            'MOVE Index': move['Value'],
            'USD 1Y IRS': irs_1y['Value'],
        }
    except Exception as e:
        logging.error(f"Error loading CSV data: {e}", exc_info=True)


def scrape_investing_data(url, name, save_path, wait=60):
    """Scrape data from Investing.com using Selenium."""
    try:
        logging.info(f"Scraping: {name} from {url}")
        logging.info("Manually select '01/01/2017' to today in the date picker and click Apply.")
        logging.info(f"Waiting {wait} seconds...")

        driver = webdriver.Chrome()  # Ensure ChromeDriver is in your PATH
        driver.get(url)
        time.sleep(wait)

        table = driver.find_element(By.TAG_NAME, "table")
        rows = table.find_elements(By.TAG_NAME, "tr")

        data = [[r.find_elements(By.TAG_NAME, 'td')[0].text.strip(),
                 r.find_elements(By.TAG_NAME, 'td')[1].text.strip().replace('%', '').replace(',', '')]
                for r in rows[1:] if len(r.find_elements(By.TAG_NAME, 'td')) >= 2]

        driver.quit()

        # Debugging: Check the scraped data
        print(data)  # Check what data is being scraped

        if not data:
            logging.error(f"No data scraped for {name}.")
            return

        df = pd.DataFrame(data, columns=["Date", "Value"])
        df["Date"] = pd.to_datetime(df["Date"], errors='coerce', dayfirst=True)
        df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
        df.dropna(subset=["Date", "Value"], inplace=True)
        df.dropna(inplace=True)
        df.set_index("Date", inplace=True)
        df.sort_index(inplace=True)
        df.to_csv(save_path)  # Ensure save_path is valid
        logging.info(f"✅ Saved {name} data to {save_path}")
    except Exception as e:
        logging.error(f"Error scraping {name} data: {e}", exc_info=True)



    # === Web Scraping ===
    # base_path = config['data']['csv_base_path']
    # scrape_investing_data(
    #     url="https://www.investing.com/indices/usd-overnight-domest-interest-rate-historical-data",
    #     name="USD Overnight Rate",
    #     save_path=f"{base_path}\\usd_overnight_rate.csv",
    #     wait=int(config['data']['scraping_wait_time'])
    # )

    # scrape_investing_data(
    #     url="https://www.investing.com/rates-bonds/usd-1-year-interest-rate-swap-historical-data",
    #     name="USD 1Y IRS",
    #     save_path=f"{base_path}\\usd_1y_irs.csv",
    #     wait=int(config['data']['scraping_wait_time'])
    # )

    # scrape_investing_data(
    #     url="https://fr.investing.com/rates-bonds/three-month-sofr-futures",
    #     name="SOFR 3M Futures",
    #     save_path=f"{base_path}\\sofr_3m_futures.csv",
    #     wait=int(config['data']['scraping_wait_time'])
    # )







# # # data_fetching.py
# import pandas as pd
# import requests
# import logging
# from ib_insync import IB, util, Stock
# from fredapi import Fred
# import configparser

# # # Existing functions unchanged

# def fetch_fmp_data(api_key, endpoint):
#     """Fetch data from FMP API."""
#     url = f"https://financialmodelingprep.com/api/v3/{endpoint}?apikey={api_key}"
#     response = requests.get(url)
#     if response.status_code == 200:
#         return response.json()
#     else:
#         logging.error(f"Error fetching data from FMP: {response.status_code} - {response.text}")
#         return None


# def get_fmp_series(config):
#     """Fetch data from FMP and return as DataFrame."""
#     try:
#         api_key = 'urJviSy2ptm8wD8ndr9O5PkA0cVLvmAV'
#         data_points = {
#             # Volatility Metrics
#             'VIX': 'quote/VIX',
#             'VXV': 'quote/VXV',
#             'MOVE Index': 'quote/MOVE',  # Ensure FMP supports this or fetch from ICE/BofA [external source]
#             'Brent Realized Volatility': 'quote/BRENT-VOL',  # Likely custom or external [external source]
#             'FX Implied Vol EURUSD': 'quote/EURUSD-IV',  # Likely Bloomberg or JPMorgan [external source]
#             'Put/Call Ratio': 'quote/PUTCALL',  # CBOE [external source]

#             # Safe-Haven Assets
#             'USD Index (DXY)': 'quote/USD',
#             'Gold Price': 'quote/GCUSD',
#             'US 10Y Treasury Yield': 'quote/10Y-Yield',

#             # Funding Stress
#             'USD Overnight Rate': 'quote/USD-Overnight-Rate',
#             'FRED RRP Volume': 'quote/FRED-RRP',
#             'TED Spread (3M LIBOR - 3M T-Bill)': 'quote/TED-Spread',  # FRED symbol: 'TEDRATE'
#             'EUR/USD Basis Swap': 'quote/EURUSD-BasisSwap',  # [external source]
#             'TONA – SOFR 2Y Swap Spread': 'quote/TONA-SOFR-SWAP-2Y',  # [external source]

#             # Valuation
#             'S&P 500 P/E Ratio': 'quote/SP500-PE'  # Could be calculated if EPS and Price available
#         }

#         fetched_data = {}
#         for key, endpoint in data_points.items():
#             result = fetch_fmp_data(api_key, endpoint)
#             if result and isinstance(result, list) and 'price' in result[0]:
#                 fetched_data[key] = result[0]['price']
#             else:
#                 logging.warning(f"Data for {key} was not fetched correctly or may require external API.")

#         df = pd.DataFrame(fetched_data, index=[pd.Timestamp.today()])
#         return df

#     except Exception as e:
#         logging.error(f"Error fetching FMP data: {e}", exc_info=True)
#         return pd.DataFrame()




