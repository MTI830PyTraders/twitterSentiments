from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
import glob, os, time


def enable_download_in_headless_chrome(driver, download_dir):
    # add missing support for chrome "send_command"  to selenium webdriver
    driver.command_executor._commands["send_command"] = ("POST", '/session/$sessionId/chromium/send_command')
    params = {'cmd': 'Page.setDownloadBehavior', 'params': {'behavior': 'allow', 'downloadPath': download_dir}}
    driver.execute("send_command", params)


def getHistoricalData(ticker):
    chrome_options = webdriver.ChromeOptions()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--allow-running-insecure-content")
    chrome_options.add_argument("--window-size=1920x1080")

    dataDirectory = os.path.dirname(os.path.realpath(__file__)) + "/data"

    # Set download dir
    chrome_options.add_experimental_option("prefs", {
        "download.default_directory": dataDirectory,
        "download.prompt_for_download": False,
        "download.directory_upgrade": True,
        "safebrowsing.enabled": True
    })

    driver = webdriver.Chrome(chrome_options=chrome_options)
    enable_download_in_headless_chrome(driver, dataDirectory)

    driver.get(
        "https://ca.finance.yahoo.com/quote/" + ticker + "/history?period1=0&period2=9999999999&interval=1d&filter=history&frequency=1d")

    if not os.path.exists("data"):
        os.makedirs("data")

    element = WebDriverWait(driver, 20).until(
        EC.element_to_be_clickable((By.XPATH, "//span[contains(text(), 'Download Data')]")))
    element.click()

    timeout = time.time() + 5  # 5 seconds from now

    # Wait for Chrome to download the file
    while not glob.glob(os.path.join(dataDirectory, "*.crdownload")):
        if time.time() > timeout:
            raise Exception("Something's wrong with the download...")
        else:
            pass
    while glob.glob(os.path.join(dataDirectory, "*.crdownload")):
        pass

    driver.close()


getHistoricalData("MSFT")
