"""
@Author: Rossi
Created At: 2023-01-18
"""

import re

import camelot
from loguru import logger
import pandas as pd
import pdfplumber
from pypdf import PdfReader, PdfWriter


def extract_financial_reports(report_file):
    pdf = pdfplumber.open(report_file)
    start_page = find_asset_report_start_page(pdf)
    asset_report = None
    if start_page == -1:
        logger.error("start page of asset report not found")
    else:
        asset_report = extract_asset_report(report_file, pdf, start_page)
    profit_report = None
    start_page = find_report_start_page_by_offset(pdf, start_page, "合并利润表")
    if start_page == -1:
        logger.error("start page of profit report not found")
    else:
        profit_report = extract_profit_report(report_file, pdf, start_page)
    cash_report = None
    start_page = find_report_start_page_by_offset(pdf, start_page, "合并现金流量表")
    if start_page == -1:
        logger.error("start page of cash report not found")
    else:
        cash_report = extract_cash_report(report_file, pdf, start_page)

    return asset_report, profit_report, cash_report


def find_asset_report_start_page(pdf):
    toc_found = False
    page_idx = None
    toc_pattern = re.compile(r"^目\s*录$")
    for i in range(10):
        page = pdf.pages[i]
        text = page.extract_text().strip()
        lines = text.split("\n")
        for line in lines[:5]:
            if toc_pattern.match(line.strip()):
                toc_found = True
                logger.debug("toc page found")
                page_idx = find_page_index(lines, ["财务报告", "财务报表", "合并资产负债表"])
                break
        if toc_found:
            break

    if page_idx is not None:
        logger.debug(f"jump to page {page_idx}")
    else:
        page_idx = 11
    for i in range(page_idx, len(pdf.pages)):
        page = pdf.pages[i]
        text = page.extract_text().strip()
        for line in text.split("\n"):
            if "合并资产负债表" in line:
                logger.debug(f"start page found: {i+1}")
                return i

    logger.warning("start page not found")

    return -1


def find_report_start_page_by_offset(pdf, offset, report_name):
    for i in range(offset, len(pdf.pages)):
        page = pdf.pages[i]
        text = page.extract_text().strip()
        for line in text.split("\n"):
            if report_name in line:
                logger.debug(f"start page found:{i+1}")
                return i

    logger.warning("startpage not found")

    return -1


def find_page_index(lines, keywords):
    pattern = re.compile(r"\d+")
    page_idx = None
    for line in lines:
        for kw in keywords:
            if kw in line and pattern.search(line) is not None:
                kw_idx = line.find(kw)
                for match in pattern.finditer(line):
                    if match.start() > kw_idx:
                        return int(match.group()) - 1
                    else:
                        page_idx = int(match.group()) - 1
                break
        if page_idx:
            break
    return page_idx


def get_clean_name(raw_name):
    if not raw_name:
        return raw_name
    clean_name = raw_name.replace("\n", "").replace(" ", "").strip()
    clean_name = clean_name.replace("/", "")
    idx = clean_name.find(":")
    if idx == -1:
        idx = clean_name.find("：")
    if idx != -1:
        if idx == len(clean_name) - 1:
            clean_name = clean_name[:-1]
        else:
            clean_name = clean_name[idx+1:]
    # pattern1 = re.compile(r"[(（]或[^(（]+[)）]")
    pattern1 = re.compile(r"[(（][^(（]+[)）]")
    pattern2 = re.compile(r"^[一二三四五六七八九十\d]+[、.]")
    # pattern4 = re.compile(r"[（(][一二三四五六七八九十]+[）)]")
    clean_name = pattern1.sub("", clean_name)
    clean_name = pattern2.sub("", clean_name)
    # clean_name = pattern3.sub("", clean_name)
    # clean_name = pattern4.sub("", clean_name)

    return clean_name.strip()


def get_refined_name(raw_name):
    if not raw_name:
        return raw_name

    refiend_name = raw_name.replace("\n", "").replace(" ", "").strip()
    return refiend_name


def refine_row(row):
    for i in range(1, len(row)-1):
        if row[i] is None or not row[i].strip():
            if row[i+1]:
                splits = re.split(r"\s+", row[i+1].strip())
                if len(splits) == 2:
                    row[i] = splits[0]
                    row[i+1] = splits[1]


def report2df(report):
    if is_header_row(report[0]):
        columns = report[0][1:]
        for i in range(len(columns)):
            columns[i] = columns[i].replace("\n", "").replace(" ", "").strip()
        index = []
        values = []
        for row in report[1:]:
            index.append(get_refined_name(row[0]))
            refine_row(row)
            values.append(row[1:])

        return pd.DataFrame(values, index=index, columns=columns)

    logger.warning("header row not found!")
    return None


def parse_finantial_report(file_path, pdf, start_page, last_item_names):
    report = []
    engine = "pdfplumber"
    flavor = "lattice"
    for i in range(start_page, len(pdf.pages)):
        if engine == "pdfplumber":
            page = pdf.pages[i]
            tables = page.extract_tables()
            if tables:
                table = tables[-1] if len(tables) > 1 and not report else tables[0]
                if len(table[0]) < 3 or len(table) < 3:
                    engine = "camelot"
                    logger.info("use camelot to extract table")
            else:
                engine = "camelot"
                logger.info("use camelot to extract table")
        if engine == "camelot":
            if flavor == "lattice":
                tables = camelot.read_pdf(file_path, pages=f"{i+1}")
                if not tables:
                    flavor = "stream"
                    logger.info("change to stream flavor")
                else:
                    table = tables[-1] if len(tables) > 1 and not report else tables[0]
                    if len(table.df.columns) < 3 or len(table.df) < 3:
                        flavor = "stream"
                        logger.info("change to stream flavor")
                    else:
                        table = table.df.values.tolist()
            if flavor == "stream":
                tables = camelot.read_pdf(file_path, flavor="stream", pages=f"{i+1}", row_tol=10, edge_tol=250)
                if not tables:
                    break
                table = tables[-1] if len(tables) > 1 and not report else tables[0]
                table = table.df.values.tolist()
        for row in table:
            if is_header_row(row) and not report:
                report.append(row)
            elif is_valid_row(row):
                report.append(row)
        if get_clean_name(report[-1][0]) in last_item_names:
            break

    return report2df(report)


def is_header_row(row):
    values = [value.strip() for value in row if value is not None]
    return ("附注" in values or "项目" in values) and ("年" in values[-1] or "日" in values[-1])


def is_valid_row(row):
    number = re.compile(r"[-]?\d+")
    has_number = False
    not_empty = 0
    for i, value in enumerate(row):
        if value is None:
            continue
        if "人" in value:
            return False
        if value.strip():
            not_empty += 1
            if i > 0 and number.match(value.strip()) and "年" not in value and "日" not in value:
                has_number = True

    return has_number and not_empty > 1


def extract_asset_report(report_file, pdf, start_page):
    report = parse_finantial_report(report_file, pdf, start_page, ["负债和股东权益总计", "负债和所有者权益总计"])
    return report


def extract_profit_report(report_file, pdf, start_page):
    report = parse_finantial_report(report_file, pdf, start_page, ["稀释每股收益"])
    return report


def extract_cash_report(report_file, pdf, start_page):
    report = parse_finantial_report(report_file, pdf, start_page, ["年末现金及现金等价物余额", "期末现金及现金等价物余额"])
    return report


def extract_financial_report_note(report_file, items, save_path):
    pdf = pdfplumber.open(report_file)
    page_num = len(pdf.pages)
    serial_number1 = re.compile(r"^([一二三四五六七八九十]+)、|\(([一二三四五六七八九十]+)\)")
    serial_number2 = re.compile(r"^\d+[.、]|\([一二三四五六七八九十]+\)")
    mapping = dict(zip("一二三四五六七八九十", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]))
    start_page = None
    end_page = None
    page_index = {}
    done = False
    serial = None
    for i in range(page_num):
        page = pdf.pages[i]
        for line in page.extract_text().strip().split("\n"):
            line = line.strip()
            if start_page is None:
                match = serial_number1.match(line)
                if match and ("合并财务报表项目" in line or "合并财务报表主要项目" in line):
                    start_page = i
                    serial = match.group(1) or match.group(2)
                    serial = mapping[serial]
                    logger.info(f"start page found: {i}")
            else:
                match = serial_number1.match(line)
                if match:
                    serial_ = match.group(1) or match.group(2)
                    serial_ = mapping.get(serial_, 0)
                    if serial_ - serial == 1 and any(kw in line for kw in ["合并范围的变更", "风险管理", "其他主体"]):
                        done = True
                        end_page = i
                        break
                if serial_number2.match(line):
                    clean_text = get_clean_name(line)
                    if clean_text in items:
                        page_index[clean_text] = i - start_page + 1
        if done:
            break

    if start_page is None:
        logger.warning("start page not found.")
        return

    pdf = PdfReader(report_file)
    pdf_writer = PdfWriter()
    for i in range(start_page, end_page+1):
        pdf_writer.add_page(pdf.pages[i])
    with open(save_path, "wb") as fo:
        pdf_writer.write(fo)

    return page_index


if __name__ == "__main__":
    # print(extract_financial_reports("./data/京东方A2021年年度报告.pdf"))
    # print(extract_financial_reports("./data/格力电器2021年年度报告.pdf"))
    # print(extract_financial_reports("data/中国平安2021年年度报告.pdf"))
    # print(extract_financial_reports("data/牧原股份2021年年度报告.pdf"))
    # import glob

    # for file in glob.glob("data/*年年度报告.pdf"):
    #     print(file)
    #     asset_report, profit_report, cash_report = extract_financial_reports(file)
    #     prefix = file.replace("年年度报告.pdf", "")
    #     asset_report.to_excel(f"{prefix}asset.xlsx")
    #     profit_report.to_excel(f"{prefix}profit.xlsx")
    #     cash_report.to_excel(f"{prefix}cash.xlsx")

    with open("data/items.txt", encoding="utf-8") as fi:
        items = set([line.strip() for line in fi])

    import glob

    for file in glob.glob1("data", "*年年度报告.pdf"):
        if file.startswith("海螺水泥"):
            continue
        print(file)
        save_path = file.replace("年年度报告", "注释")
        page_index = extract_financial_report_note(f"data/{file}", items, save_path)
        print(len(page_index))
