import base64
from collections import Counter
import math
import os
import re

import camelot
import fitz
from loguru import logger
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer, LTTextLineHorizontal
import pdfplumber
from pypdf import PdfReader, PdfWriter


section_title = re.compile(r"(?P<section_name>第\s*(\d+|[一二三四五六七八九十]+)\s*节.+)")


def parse_toc(pdf_file):
    pdf = pdfplumber.open(pdf_file)
    toc_pattern = re.compile(r"^目\s*录$")
    toc_page = None
    for i in range(10):
        page = pdf.pages[i]
        text = page.extract_text().strip()
        lines = text.split("\n")
        for line in lines[:5]:
            if toc_pattern.match(line.strip()):
                toc_page = page
                break
        if toc_page:
            break
    if not toc_page:
        logger.warning("toc page not found")
        return []

    p1 = re.compile(r"(?P<section_title>^[^\d]+?\s*[^.]*)[.]+\s*\d+$")
    p2 = re.compile(r"^\d+\s+(?P<section_title>[^\d]+)$")
    pattern = None
    section_titles = []
    for line in page.extract_text().strip().split("\n"):
        line = line.strip()
        if pattern is None:
            if p1.match(line):
                pattern = p1
            elif p2.match(line):
                pattern = p2
        if pattern:
            match = pattern.match(line)
            if match:
                title = match.group("section_title")
                section_titles.append(title.strip().replace(" ", ""))

    return section_titles


def segment_by_section(pdf_file, section_titles, save_dir=None):
    idx = 0
    start = None
    section = None
    sections = {}
    serial_pattern = re.compile("第[一二三四五六七八九十]+[节章]")
    extra_pattern = re.compile(r"[(（].+[)）]$")
    pdf = pdfplumber.open(pdf_file)
    for i, page in enumerate(pdf.pages):
        for line in page.extract_text().strip().split("\n"):
            line = line.strip().replace(" ", "")
            line = extra_pattern.sub("", line)
            title = section_titles[idx]
            clean_title = serial_pattern.sub("", title)
            clean_title = extra_pattern.sub("", clean_title)
            if line == title or clean_title == line:
                if start is not None:
                    sections[section_titles[idx-1]] = (start, i)
                start = i
                idx += 1
                if idx == len(section_titles):
                    break
        if idx == len(section_titles):
            break
    if start:
        sections[section_titles[-1]] = (start, len(pdf.pages)-1)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        pdf = PdfReader(pdf_file)
        for section, (start, end) in sections.items():
            writer = PdfWriter()
            for i in range(start, end+1):
                writer.add_page(pdf.pages[i])
            file_name = section.replace(" ", "")
            writer.write(f"{save_dir}/{file_name}.pdf")


def seperate_financial_report(pdf_file, save_dir):
    reader = PdfReader(pdf_file)
    writer = PdfWriter()
    idx = 0
    done = False
    while idx < len(reader.pages):
        for line in reader.pages[idx].extract_text().strip().split("\n"):
            if any(line.strip().endswith(suffix) for suffix in ["公司基本情况", "集团基本情况"]):
                writer.write(f"{save_dir}/财务报告.pdf")
                writer = PdfWriter()
                for i in range(idx, len(reader.pages)):
                    writer.add_page(reader.pages[i])
                writer.write(f"{save_dir}/财务报表附注.pdf")
                done = True
                break
        else:
            writer.add_page(reader.pages[idx])
            idx += 1
        if done:
            break


def get_sorted_lines(page):
    lines = []
    for element in page:
        if isinstance(element, LTTextContainer):
            for text_line in element:
                if isinstance(text_line, LTTextLineHorizontal):
                    lines.append(text_line)
    lines = [line for line in sorted(lines, key=lambda x: x.bbox[3], reverse=True)]

    return lines


def get_most_common(positions):
    if not positions:
        return 0, 0
    counts = Counter(positions)
    top2 = counts.most_common(2)
    if len(top2) == 1:
        return top2[0]
    else:
        if abs(top2[0][0] - top2[1][0]) == 1:
            count = top2[0][1] + top2[1][1]
        else:
            count = top2[0][1]
        return top2[0][0], count


def extract_head_foot_note(pdf_file):
    tops = []
    bottoms = []
    for i, page in enumerate(extract_pages(pdf_file)):
        height = page.bbox[3]
        lines = get_sorted_lines(page)
        for j in range(len(lines)-1):
            margin = lines[j].bbox[1] - lines[j+1].bbox[3]
            if margin > 20:
                tops.append(math.floor(height - lines[j].bbox[1]))
                break
        for j in range(len(lines)-1, 0, -1):
            margin = lines[j-1].bbox[1] - lines[j].bbox[3]
            if margin > 25:
                bottoms.append(math.ceil(lines[j].bbox[3]))
                break
        if i == 10:
            break
    top, count = get_most_common(tops)
    if count < 5:
        top = None
    bottom, count = get_most_common(bottoms)
    if count < 5:
        bottom = None
    return top, bottom


# def extract_head_foot_note(pdf_file, head_lines=1, foot_lines=1):
#     pages = extract_pages(pdf_file)
#     page = next(pages)
#     lines = get_sorted_lines(page)
#     width, height = page.bbox[2], page.bbox[3]
#     top = lines[head_lines-1].bbox[1]
#     bottom = lines[-foot_lines].bbox[-1]
#     return [(0, 0, width, bottom), (0, top, width, height)]


def curves_to_edges(cs):
    """See https://github.com/jsvine/pdfplumber/issues/127"""
    edges = []
    for c in cs:
        edges += pdfplumber.utils.rect_to_edges(c)
    return edges


def extract_text_without_table_content(pdf, page_index):
    page = pdf.pages[page_index]
    # Table settings.
    ts = {
        "vertical_strategy": "explicit",
        "horizontal_strategy": "explicit",
        "explicit_vertical_lines": curves_to_edges(page.curves + page.edges),
        "explicit_horizontal_lines": curves_to_edges(page.curves + page.edges),
        "intersection_y_tolerance": 10,
    }

    bboxes = [table.bbox for table in page.find_tables(table_settings=ts)]

    def not_within_bboxes(obj):
        top = page.chars[0]["bottom"]
        bottom = page.chars[-1]["top"]

        def obj_in_bbox(_bbox):
            """See https://github.com/jsvine/pdfplumber/blob/stable/pdfplumber/table.py#L404"""
            v_mid = (obj["top"] + obj["bottom"]) / 2
            h_mid = (obj["x0"] + obj["x1"]) / 2
            x0, top, x1, bottom = _bbox
            return (h_mid >= x0) and (h_mid < x1) and (v_mid >= top) and (v_mid < bottom)
        return obj["top"] >= top and obj["bottom"] <= bottom and not any(obj_in_bbox(__bbox) for __bbox in bboxes)

    return page.filter(not_within_bboxes).extract_text()


def recoverpix(doc, item):
    xref = item[0]  # xref of PDF image
    smask = item[1]  # xref of its /SMask

    # special case: /SMask or /Mask exists
    if smask > 0:
        pix0 = fitz.Pixmap(doc.extract_image(xref)["image"])
        if pix0.alpha:  # catch irregular situation
            pix0 = fitz.Pixmap(pix0, 0)  # remove alpha channel
        mask = fitz.Pixmap(doc.extract_image(smask)["image"])

        try:
            pix = fitz.Pixmap(pix0, mask)
        except Exception:  # fallback to original base image in case of problems
            pix = fitz.Pixmap(doc.extract_image(xref)["image"])

        if pix0.n > 3:
            ext = "pam"
        else:
            ext = "png"

        return {  # create dictionary expected by caller
            "ext": ext,
            "colorspace": pix.colorspace.n,
            "image": pix.tobytes(ext),
        }

    # special case: /ColorSpace definition exists
    # to be sure, we convert these cases to RGB PNG images
    if "/ColorSpace" in doc.xref_object(xref, compressed=True):
        pix = fitz.Pixmap(doc, xref)
        pix = fitz.Pixmap(fitz.csRGB, pix)
        return {  # create dictionary expected by caller
            "ext": "png",
            "colorspace": 3,
            "image": pix.tobytes("png"),
        }
    return doc.extract_image(xref)


def extract_images(pdf_file, page_idx):
    pdf = fitz.open(pdf_file)
    page = pdf[page_idx]
    image_list = pdf.get_page_images(page_idx, full=True)
    images = []
    for image in image_list:
        bbox = page.get_image_bbox(image)
        image = recoverpix(pdf, image)
        image["bbox"] = bbox
        images.append(image)

    return images


def extract_tables(pdf_file, page_idx, kwargs={}):
    tables = camelot.read_pdf(pdf_file, pages=f"{page_idx+1}", **kwargs)

    return tables


def convert_bbox(bbox, height):
    x0, y0, x1, y1 = bbox
    return (x0, height-y1, x1, height-y0)


def is_in_bbox(obj, bbox):
    v_mid = (obj.bbox[1] + obj.bbox[3]) / 2
    h_mid = (obj.bbox[0] + obj.bbox[2]) / 2
    x0, top, x1, bottom = bbox

    return (h_mid >= x0) and (h_mid < x1) and (v_mid >= top) and (v_mid < bottom)


def extract_lines_not_within_bboxes(page, bboxes):
    lines = get_sorted_lines(page)
    lines = [line for line in lines if not any(is_in_bbox(line, bbox) for bbox in bboxes)]

    return lines


def extract_page_elements(pdf_file, page_idx, top, bottom, table_settings={}):
    pages = extract_pages(pdf_file, page_numbers=[page_idx])
    page = next(pages)
    height = page.bbox[3]
    tables = extract_tables(pdf_file, page_idx, table_settings)
    images = extract_images(pdf_file, page_idx)
    elements = [(table, table._bbox) for table in tables] + [(image, convert_bbox(image["bbox"], height)) for image in images]
    bboxes = []
    if top is not None:
        bboxes.append((0, page.bbox[3]-top, page.bbox[2], page.bbox[3]))
    if bottom is not None:
        bboxes.append((0, 0, page.bbox[2], bottom))
    bboxes.extend([element[1] for element in elements])
    lines = extract_lines_not_within_bboxes(page, bboxes)
    elements.extend((line, line.bbox) for line in lines)
    elements = sorted(elements, key=lambda x: x[1][3], reverse=True)
    elements = [element[0] for element in elements]

    return elements


char2digit = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "零": 0,
              "壹": 1, "贰": 2, "叁": 3, "肆": 4, "伍": 5, "陆": 6, "柒": 7, "捌": 8, "玖": 9, "两": 2,
              "": 1}
digit = "一二三四五六七八九十两"
number = re.compile(f"^((?P<bai>[{digit}]+)[百佰])?((?P<shi>[{digit}]*)[十拾])?(?P<ge>[{digit}]+)?$")


def str2number(num_str):
    match = number.match(num_str)
    if not match:
        return 0
    return char2digit.get(match.group("bai"), 0) * 100 + char2digit.get(match.group("shi"), 0) * 10 + char2digit.get(match.group("ge"), 0)


serial_numbers = [r"(?P<number>[一二三四五六七八九十]+)([、.]|\s).+", r"(?P<number>\d+)\.[^\d]+", r"[(（](?P<number>[一二三四五六七八九十]+)[)）].+",
                  r"[(（]?(?P<number>\d+)[)）][.|\s].+", r"(?P<number>\d+)、.+"]
patterns = [re.compile(item) for item in serial_numbers]


def segment_content(content):
    content = content.strip()

    lines = content.split("\n")
    stack = []
    current_serial_number = None
    current_pattern = None
    text = ""
    parts = []

    for line in lines:
        matched = False
        for pattern in patterns:
            match = pattern.match(line)
            if match:
                matched = True
                serial_number = match.group("number")
                if serial_number.isdigit():
                    serial_number = int(serial_number)
                else:
                    serial_number = str2number(serial_number)
                if pattern is not current_pattern:
                    num = match.group("number")
                    if (num == "1" or num == "一"):
                        if text:
                            parts.append(text)
                        text = line + "\n"
                        if current_pattern is not None:  # 下钻
                            stack.append((parts, current_pattern, current_serial_number))
                            parts = []
                        current_pattern = pattern
                        current_serial_number = serial_number
                    elif stack and stack[-1][-1] == serial_number - 1:  # 上钻
                        parts.append(text)
                        text = line + "\n"
                        sub_parts = parts
                        parts, current_pattern, current_serial_number = stack.pop()
                        parts[-1] = [parts[-1], sub_parts]
                        current_serial_number = serial_number
                    else:
                        matched = False
                else:
                    if serial_number - current_serial_number != 1:
                        matched = False
                    else:
                        parts.append(text)
                        text = line + "\n"
                        current_serial_number = serial_number
                break

        if not matched:
            text += line + "\n"

    parts.append(text)
    if stack:
        sub_parts = parts
        parts, current_pattern, _ = stack.pop()
        parts[-1] = [parts[-1], sub_parts]
    return parts


def segment_paragraph(lines):
    paragraphs = []
    para = ""
    prev_len = None
    indent = None
    max_len = max(line.bbox[2] - line.bbox[0] for line in lines)
    for i, line in enumerate(lines):
        length = line.bbox[2] - line.bbox[0]
        text = line.get_text().strip()
        if i > 0:
            indent = line.bbox[0] - lines[i-1].bbox[0]
        if indent is not None and indent > 15:
            if para:
                paragraphs.append(para)
            if length < max_len * 0.9:
                paragraphs.append(text)
                para = ""
                prev_len = None
            else:
                para = text
                prev_len = length
        elif prev_len is not None:
            if any(pattern.match(text) for pattern in patterns):
                paragraphs.append(para)
                if length < max_len * 0.9:
                    paragraphs.append(text)
                    prev_len = None
                    para = ""
                else:
                    prev_len = length
                    para = text
            elif length < 0.9 * prev_len:
                para += text
                paragraphs.append(para)
                para = ""
                prev_len = None
            else:
                para += text
                prev_len = length
        else:
            if length < max_len * 0.8:
                paragraphs.append(text)
            else:
                para = text
                prev_len = length

    if para:
        paragraphs.append(para)

    return paragraphs


def arrange_elements(elements):
    lines = [element["line"] for element in elements if "line" in element]
    paragraphs = segment_paragraph(lines)
    para_idx = -1
    offset = 0
    new_elements = []
    for element in elements:
        if "line" in element:
            if para_idx == -1:
                para_idx += 1
                new_elements.append({"text": paragraphs[para_idx]})
                continue
            text = element["line"].get_text().strip()
            paragraph = paragraphs[para_idx]
            idx = paragraph.find(text, offset)
            if idx == -1:
                para_idx += 1
                paragraph = paragraphs[para_idx]
                idx = paragraph.find(text)
                assert idx > -1
                new_elements.append({"text": paragraph})
                offset = idx + len(text)
            else:
                offset = idx + len(text)
        else:
            new_elements.append(element)

    return new_elements


def trim_extra_content(pages):
    if len(pages) > 1:
        for i, element in enumerate(pages[-1]):
            if isinstance(element, LTTextLineHorizontal):
                if section_title.search(element.get_text()):
                    pages[-1] = pages[-1][:i]
                    if not pages[-1]:
                        pages.pop()
                    break
    else:
        for i, element in enumerate(pages[0]):
            if isinstance(element, LTTextLineHorizontal):
                if section_title.search(element.get_text()):
                    pages[0] = pages[0][:i]
                    break


def extract_pdf_elements(pdf_file, table_settings={}):
    pdf = pdfplumber.open(pdf_file)
    page_num = len(pdf.pages)
    pdf.close()
    pages = []
    top, bottom = extract_head_foot_note(pdf_file)
    logger.info("start to extract page elements")
    for i in range(page_num):
        elements = extract_page_elements(pdf_file, i, top, bottom, table_settings)
        pages.append(elements)
        if (i + 1) % 5 == 0:
            logger.info(f"{i+1} pages processed")

    trim_extra_content(pages)

    all_elements = []
    for page in pages:
        for i, element in enumerate(page):
            if isinstance(element, LTTextLineHorizontal):
                all_elements.append({"line": element})
            elif isinstance(element, dict):
                all_elements.append(element)
            else:
                table = {"table": element}
                if all_elements:
                    if "line" in all_elements[-1]:
                        text = all_elements[-1]["line"].get_text()
                        if "单位:" in text or "单位：" in text:
                            head_note = text.strip()
                            all_elements.pop()
                            table["head_note"] = head_note
                    elif i == 0 and "table" in all_elements[-1]:
                        prev_table = all_elements[-1]["table"]
                        if len(element.df.columns) == len(prev_table.df.columns):
                            prev_table.df = prev_table.df.append(element.df)
                            table = None
                if table:
                    all_elements.append(table)
    elements = arrange_elements(all_elements)

    return elements


def pdf2html(pdf_file, save_file, table_settings={}):
    elements = extract_pdf_elements(pdf_file, table_settings)
    with open("pdf.html", encoding="utf-8") as fi:
        html = fi.read()
    content = ""
    for element in elements:
        if "text" in element:
            content += f"<p>{element['text']}</p>\n"
        elif "image" in element:
            uri = base64.b64encode(element["image"]).decode("utf-8")
            content += f'<div align="center"><img src="data:image/jpeg;base64,{uri}"></div>'
        else:
            if "head_note" in element:
                content += f'<div class="head-note"><span>{element["head_note"]}</span></div>'
            table = element["table"]
            content += '<div class="table-container">' + table.df.to_html(header=False, index=False).replace("\\n", "") + '</div>'
            content += "<br/>"
    html = html.replace("{{content}}", content)
    with open(save_file, "w", encoding="utf-8") as fo:
        fo.write(html)


if __name__ == "__main__":
    # pdf = pdfplumber.open("./data/招商银行招商银行股份有限公司2022年度报告.pdf")
    # print(pdf.pages[76].extract_text())
    # for page in extract_pages("./data/中国平安中国平安2022年年度报告.pdf", page_numbers=[3]):
    #     for line in get_sorted_lines(page):
    #         print(line.get_text())

    # titles = parse_toc("data/招商银行招商银行股份有限公司2022年度报告.pdf")
    # segment_by_section("data/招商银行招商银行股份有限公司2022年度报告.pdf", titles, "data/zhaohang")

    # titles = parse_toc("data/上海机场上海机场2022年年度报告.pdf")
    # segment_by_section("data/上海机场上海机场2022年年度报告.pdf", titles, "data/shangji")

    # titles = parse_toc("data/格力电器2022年年度报告.pdf")
    # segment_by_section("data/格力电器2022年年度报告.pdf", titles, "data/geli")

    # segment_by_section("data/中国平安中国平安2022年年度报告.pdf", ["释义", "客户经营分析", "公司治理报告", "审计报告", "平安大事记"], "data/pingan")

    # for file in os.listdir("data"):
    #     if file.endswith(".pdf"):
    #         print(file)
    #         save_dir = f"data/{file[:4]}"
    #         os.makedirs(save_dir, exist_ok=True)
    #         segment_by_section(f"./data/{file}", save_dir)
    # seperate_financial_report("data/第十节财务报告.pdf", "data")

    # for file in os.listdir("data"):
    #     file = file.lower()
    #     if file.startswith("第") and file.endswith(".pdf") and "财务" not in file:
    #         print(file)
    #         save_file = file.replace(".pdf", ".html")
    #         pdf2html(f"./data/{file}", f"data/{save_file}")

    # tables = camelot.read_pdf("./data/第二节公司简介和主要财务指标.pdf", pages="4", line_scale=35)
    # print(tables)
    # print(tables[-1])
    # print(tables[0].df)
    # for table in tables:
    #     print(table._bbox)
    # print(table.df.to_html(header=False, index=False))

    # reader = PdfReader("./data/第三节管理层讨论与分析.pdf")
    # writer = PdfWriter()
    # writer.add_page(reader.pages[4])
    # writer.write("temp.pdf")

    # prev = None
    # for page in extract_pages("temp.pdf"):
    #     for element in page:
    #         if isinstance(element, LTTextContainer):
    #             for line in element:
    #                 if isinstance(line, LTTextLineHorizontal):
    #                     if prev is not None:
    #                         indent = line.bbox[0] - prev.bbox[0]
    #                         if indent > 1:
    #                             print(line.get_text().strip(), indent, line.bbox[2])
    #                     prev = line

    # print(extract_head_foot_note("data/贵州茅台贵州茅台2022年年度报告.pdf"))
    # print(extract_head_foot_note("data/招商银行招商银行股份有限公司2022年度报告.pdf"))
    # print(extract_head_foot_note("data/中国平安中国平安2022年年度报告.pdf"))
    # print(extract_head_foot_note("data/上海机场上海机场2022年年度报告.pdf"))
    # print(extract_head_foot_note("data/分众传媒2022年年度报告.pdf"))

    # pdf2html("data/geli/第一节重要提示、目录和释义.pdf", "data/geli/shiyi.html")
    # pdf2html("data/geli/第二节公司简介和主要财务指标.pdf", "data/geli/jianjie.html")
    # pdf2html("data/geli/第三节管理层讨论与分析.pdf", "data/geli/taolun.html")
    # pdf2html("data/geli/第四节公司治理.pdf", "data/geli/zhili.html")
    # pdf2html("data/geli/第五节环境和社会责任.pdf", "data/geli/zeren.html")
    # pdf2html("data/geli/第六节重要事项.pdf", "data/geli/shixiang.html")
    # pdf2html("data/geli/第七节股份变动及股东情况.pdf", "data/geli/biandong.html")
    # table_settings = {"line_scale": 35, "strip_text": " \n"}
    table_settings = {"flavor": "stream", "row_tol": 10, "edge_tol": 250}
    pdf2html("data/geli/第十节财务报告.pdf", "data/geli/caiwu.html", table_settings)
