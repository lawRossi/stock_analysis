"""
@Author: Rossi
Created At: 2023-01-08
"""

import json
import os
import random
import re
import time
import traceback

import akshare as ak
from loguru import logger
from lxml.etree import HTML
import pandas as pd
import requests
import tushare as ts


# tushare 用户名：1142595193@qq.com 密码 luo0322
token = "950e1fcbbaf085a587b776a1e1be13d1ba722d5f9dae72c627af323a"


def get_stock_list():
    url = "http://api.tushare.pro"
    data = {
        "token": token,
        "api_name": "stock_basic",
        "params": {"list_status": "L"},
        "fields": "ts_code,name,industry,market,list_date,list_status"
    }
    res = requests.post(url, json=data).json()
    fields = res["data"]["fields"]
    for item in res["data"]["items"]:
        stock = dict(zip(fields, item))
        stock["code"] = stock["ts_code"][:stock["ts_code"].find(".")]
        stock["exchange"] = stock["ts_code"][stock["ts_code"].find(".")+1:]
        yield stock


def get_company_info(stock_code):
    pro = ts.pro_api(token)
    data = pro.stock_company(ts_code=stock_code, fields="chairman,manager,secretary,reg_capital,setup_date,province,city,introduction,employees,main_business,business_scope")
    info = {}
    for column in data.columns:
        info[column] = data.loc[0, column]
    return info


def get_stock_info(stock_name_or_code):
    res = requests.get(f"https://searchapi.eastmoney.com/api/suggest/get?input={stock_name_or_code}&type=14&token=D43BF722C8E33BDC906FB84D85E326E8").json()
    print(res)

    if "QuotationCodeTable" in res:
        data = res["QuotationCodeTable"]["Data"][0]
        stock_code = data["Code"]
        stock_name = data["Name"]
        security_name = data["SecurityTypeName"]
        if security_name == "沪A":
            exchange = "SH"
        elif security_name == "深A":
            exchange = "SZ"
        elif security_name == "港股":
            exchange = "HK"
        else:
            exchange = "US"
        return {"stock_name": stock_name, "stock_code": stock_code, "exchange": exchange}
    return None


def download_financial_reports(stock_code, save_dir):
    save_dir = os.path.join(save_dir, stock_code)
    os.makedirs(save_dir, exist_ok=True)

    headers = {
        "Referer": "https://finance.sina.com.cn/",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.163 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9"
    }

    url = f"https://money.finance.sina.com.cn/corp/go.php/vDOWN_BalanceSheet/displaytype/4/stockid/{stock_code}/ctrl/all.phtml"
    save_path = os.path.join(save_dir, "asset_reports.csv")
    try:
        content = requests.get(url, headers=headers).content
        with open(save_path, "wb") as fo:
            fo.write(content)
        logger.info(f"finished downloading asset reports for {stock_code}")
    except Exception:
        traceback.print_exc()
        logger.warning(f"fail to download asset reports for {stock_code}")

    url = f"https://money.finance.sina.com.cn/corp/go.php/vDOWN_ProfitStatement/displaytype/4/stockid/{stock_code}/ctrl/all.phtml"
    save_path = os.path.join(save_dir, "profit_reports.csv")
    try:
        content = requests.get(url, headers=headers).content
        with open(save_path, "wb") as fo:
            fo.write(content)
        logger.info(f"finished downloading profit reports for {stock_code}")
    except Exception:
        logger.warning(f"fail to download profit reports for {stock_code}")

    url = f"https://money.finance.sina.com.cn/corp/go.php/vDOWN_CashFlow/displaytype/4/stockid/{stock_code}/ctrl/all.phtml"
    save_path = os.path.join(save_dir, "cash_reports.csv")
    try:
        content = requests.get(url, headers=headers).content
        with open(save_path, "wb") as fo:
            fo.write(content)
        logger.info(f"finished downloading cash reports for {stock_code}")
    except Exception:
        logger.warning(f"fail to download cash reports for {stock_code}")


def crawl_financial_reports():
    stocks = list(get_stock_list())
    for stock in stocks[100:]:
        if stock["list_status"] == "L":
            code = stock["code"]
            logger.info(f"start to crawl {code}")
            try:
                download_financial_reports(code, "data/financial_report")
                time.sleep(random.randint(10, 20))
            except Exception:
                logger.warning(f"error occurs when crawling {code}")
    logger.info("finished crawling !!!")


def get_notices(stock_code, page):
    url = "http://np-anotice-stock.eastmoney.com/api/security/ann?page_size=50&page_index=%s&ann_type=A&client_source=web&stock_list=%s" % (page, stock_code)
    text = requests.get(url).text
    content = json.loads(text)
    return content["data"]["list"]


def get_pdf_url(art_code):
    url = "http://np-cnotice-stock.eastmoney.com/api/content/ann?art_code=%s&client_source=web&page_index=1" % art_code
    res = requests.get(url).json()
    return res["data"]["attach_url_web"]


def get_annual_report(stock_code, n):
    for i in range(3*n):
        try:
            notices = get_notices(stock_code, i+1)
            for notice in notices:
                title = notice["title"]
                unrelated = ["摘要", "英文", "补充", "披露", "公告", "问询", "意见", "董事", "通知"]
                if ("年度报告" in title or "年报" in title) and not any(word in title for word in unrelated):
                    yield (notice["title"], notice["art_code"])
            time.sleep(2)
        except Exception:
            logger.warning(f"fail to get notices for {stock_code}, page {i}")


def download_annual_reports(stock_code, save_dir, n):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    downloaded = 0
    for item in get_annual_report(stock_code, n):
        logger.info(item[0])
        download_report(item[0], item[1], save_dir)
        if item[0].endswith("年度报告") or item[0].endswith("年报"):
            downloaded += 1
        if downloaded == n:
            break
        time.sleep(3)


def download_report(title, art_code, save_dir):
    try:
        link = get_pdf_url(art_code)
        headers = {
            "Referer": link,
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.163 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9",
            "Accept-Encoding": "gzip, deflate, br"
        }

        file_name = re.sub("^\d+", "", title).replace(":", "") + ".pdf"
        save_path = os.path.join(save_dir, file_name)
        with open(save_path, "wb") as fo:
            fo.write(requests.get(link, headers=headers).content)
    except Exception:
        logger.warning(f"fail to download {title}")


def crawl_annual_reports(num):
    for stock in get_stock_list():
        if stock["list_status"] == "L":
            code = stock["code"]
            logger.info(f"start to crawl {code}")
            try:
                download_annual_reports(code, f"data/annual_report/{code}/", num)
                time.sleep(5)
            except Exception:
                logger.warning(f"error occurs when crawling {code}")


def get_recent_info(stock_code_or_name):
    info = get_stock_info(stock_code_or_name)
    if info["exchange"] == "SZ":
        secid = f"0.{info['stock_code']}"
    else:
        secid = f"1.{info['stock_code']}"
    url = f"https://push2.eastmoney.com/api/qt/stock/get?fltt=2&invt=2&secid={secid}&fields=f43,f116,f117,f162,f167,f163,f164"
    data = requests.get(url).json()
    data = data["data"]
    return {
        "股票代码": info["stock_code"],
        "股票名称": info["stock_name"],
        "总市值": data["f116"],
        "流通市值": data["f117"],
        "市盈率(动)": data["f162"],
        "市盈率(静)": data["f163"],
        "市盈率(TTM)": data["f164"],
        "市净率": data["f167"],
        "现价": data["f43"]
    }


def get_financial_index_report(stock_code, year):
    headers = {
        "Referer": "https://finance.sina.com.cn/",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.163 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9"
    }
    url = f"http://money.finance.sina.com.cn/corp/go.php/vFD_FinancialGuideLine/stockid/{stock_code}/ctrl/{year}/displaytype/4.phtml"
    try:
        res = requests.get(url, headers=headers)
        html = HTML(res.text)
        table = html.xpath('//table[@id="BalanceSheetNewTable0"]')[0]
        columns = []
        index = []
        values = []
        category = None
        datas = {}
        for tr in table.xpath('.//tr'):
            tds = tr.xpath("./td")
            if len(tds) == 1:
                texts = tds[0].xpath("./strong/text()")
                if texts:
                    if category:
                        datas[category] = pd.DataFrame(values, index=index, columns=columns)
                        index = []
                        values = []
                    category = texts[0]
            elif len(tds) == 4 or len(tds) == 5:
                text = tds[0].xpath(".//text()")[0].strip()
                if text == "报告日期":
                    for td in tds[1:]:
                        columns.append(td.xpath(".//text()")[0].strip())
                else:
                    index.append(tds[0].xpath(".//text()")[0].strip())
                    row = []
                    for td in tds[1:]:
                        value = td.xpath(".//text()")[0].strip()
                        if value == "--":
                            row.append(None)
                        else:
                            row.append(float(value))
                    values.append(row)
        if category:
            datas[category] = pd.DataFrame(values, index=index, columns=columns)

        return datas

    except Exception:
        traceback.print_exc()
        logger.warning(f"fail to get financial index for {stock_code} year {year}")


def download_financial_index_reports(code, years, save_dir):
    all_datas = None
    for year in years:
        datas = get_financial_index_report(code, year)
        if not all_datas:
            all_datas = datas
        elif datas is not None:
            for key, data in datas.items():
                all_datas[key] = pd.concat([all_datas[key], data], axis=1)
    if all_datas is None:
        return

    os.makedirs(os.path.join(save_dir, code), exist_ok=True)
    save_path = os.path.join(save_dir, code, "financial_index_reports.xlsx")
    with pd.ExcelWriter(save_path) as writer:
        for category, data in all_datas.items():
            data.to_excel(writer, sheet_name=category)


def crawl_financial_index_reports():
    stocks = list(get_stock_list())
    for stock in stocks[100:]:
        if stock["list_status"] == "L":
            code = stock["code"]
            logger.info(f"start to crawl {code}")
            try:
                years = ["2022", "2021", "2020", "2019", "2018", "2017", "2016"]
                download_financial_index_reports(code, years, "data/financial_report")
                time.sleep(random.randint(10, 20))
            except Exception:
                logger.warning(f"error occurs when crawling {code}")
    logger.info("finished crawling !!!")


def get_history_prices(stock_code):
    url = "http://52.push2his.eastmoney.com/api/qt/stock/kline/get?"
    payload = {
        "cb": f"jQuery35102775669884046077_{str(time.time()).replace('.', '')[:13]}",
        "secid": f"0.{stock_code}",
        "ut": "fa5fd1943c7b386f172d6893dbfba10b",
        "fields1": "f1%2Cf2%2Cf3%2Cf4%2Cf5%2Cf6",
        "fields2": "f51%2Cf52%2Cf53%2Cf54%2Cf55%2Cf56%2Cf57%2Cf58%2Cf59%2Cf60%2Cf61",
        "klt": "102",
        "fqt": "0",
        "end": "20500101",
        "lmt": "1000000"
    }
    for k, v in payload.items():
        url += f"{k}={v}&"
    url = url[:-1]
    headers = {
        "Referer": "http://quote.eastmoney.com/",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.163 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9"
    }

    try:
        text = requests.get(url, headers=headers, timeout=5).text
        p = re.compile(r"\((\{.+\})\)")
        match = p.search(text)
        if match:
            data = json.loads(match.groups(1)[0])
            klines = data["data"]["klines"]
            dates = []
            rows = []
            keys = ["开盘价", "收盘价", "最高价", "最低价", "成交量", "成交额", "振幅", "涨跌幅", "涨跌额", "换手率"]
            for kline in klines:
                splits = kline.split(",")
                dates.append(splits[0])
                rows.append(dict(zip(keys, splits[1:])))
            data = pd.DataFrame(rows, index=dates)

            return data
    except Exception:
        logger.warning(f"fail to get history prices for {stock_code}")
        return None


if __name__ == "__main__":
    # crawl_annual_reports(20)
    # crawl_financial_index_reports()
    # crawl_financial_reports()
    # print(get_company_info("000001.SZ"))
    # print(get_history_prices("000001"))
    # download_annual_reports("600519", "data", 2)
    # download_financial_reports("600519", "data/financial_report")

    stock_zygc_em_df = ak.stock_zygc_ym(symbol="600009")
    # stock_zygc_em_df = ak.stock_zyjs_ths(symbol="SH600009")
    stock_zygc_em_df.to_excel("temp.xlsx", index=False)
