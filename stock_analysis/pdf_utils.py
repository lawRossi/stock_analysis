import base64
from collections import Counter
import math
import os
import re

import camelot
import fitz
from loguru import logger
import pandas as pd
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


class Table:
    def __init__(self):
        self.bbox = None
        self.df = None


def extract_tables(pdf_file, page_idx, kwargs={}):
    tables = []
    engine = kwargs["engine"]
    kwargs_ = {k: v for k, v in kwargs.items() if k != "engine"}
    if engine == "camelot":
        for raw_table in camelot.read_pdf(pdf_file, pages=f"{page_idx+1}", **kwargs_):
            table = Table()
            table.bbox = raw_table._bbox
            table.df = raw_table.df
            tables.append(table)
    elif engine == "pdfplumber":
        pdf = pdfplumber.open(pdf_file)
        page = pdf.pages[page_idx]
        for raw_table in page.find_tables(**kwargs_):
            table = Table()
            table.bbox = convert_bbox(raw_table.bbox, page.height)
            table.df = pd.DataFrame(raw_table.extract()).fillna("")
            tables.append(table)

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


def extract_all_images(pdf_file):
    pdf = fitz.open(pdf_file)
    all_images = []
    for i in range(len(pdf)):
        page = pdf[i]
        image_list = pdf.get_page_images(i, full=True)
        images = []
        for image in image_list:
            bbox = page.get_image_bbox(image)
            image = recoverpix(pdf, image)
            image["bbox"] = bbox
            images.append(image)
        all_images.append(images)

    return all_images


def get_page_number(pdf_file):
    with open(pdf_file, "rb") as fi:
        pdf = PdfReader(fi)
        return len(pdf.pages)


def extract_all_tables(pdf_file, kwargs):
    engine = kwargs["engine"]
    kwargs_ = {k: v for k, v in kwargs.items() if k != "engine"}

    if engine == "camelot":
        page_num = get_page_number(pdf_file)
        all_tables = [[] for _ in range(page_num)]
        for raw_table in camelot.read_pdf(pdf_file, pages="all", **kwargs_):
            table = Table()
            table.bbox = raw_table._bbox
            table.df = raw_table.df
            all_tables[raw_table.parsing_report["page"]-1].append(table)

    elif engine == "pdfplumber":
        pdf = pdfplumber.open(pdf_file)
        all_tables = []
        for page in pdf.pages:
            tables = []
            for raw_table in page.find_tables(**kwargs_):
                table = Table()
                table.bbox = convert_bbox(raw_table.bbox, page.height)
                table.df = pd.DataFrame(raw_table.extract()).fillna("")
                tables.append(table)
            all_tables.append(tables)

    return all_tables


def extract_all_page_elements(pdf_file, top, bottom, table_settings={}):
    pages = extract_pages(pdf_file)
    all_images = extract_all_images(pdf_file)
    all_tables = extract_all_tables(pdf_file, table_settings)
    all_elements = []
    for page, images, tables in zip(pages, all_images, all_tables):
        height = page.bbox[3]
        elements = [(table, table.bbox) for table in tables] + [(image, convert_bbox(image["bbox"], height)) for image in images]
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
        all_elements.append(elements)

    return all_elements


char2digit = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "零": 0,
              "壹": 1, "贰": 2, "叁": 3, "肆": 4, "伍": 5, "陆": 6, "柒": 7, "捌": 8, "玖": 9, "两": 2,
              "": 1, "①": 1, "②": 2, "③": 3, "④": 4, "⑤": 5, "⑥": 6, "⑦": 7, "⑧": 8, "⑨": 9, "⑩": 10,
              "⑪": 11, "⑫": 12, "⑬": 13, "⑭": 14, "⑮": 15, "⑯": 16, "⑰": 17, "⑱": 18, "⑲": 19, "⑳": 20}
digit = "一二三四五六七八九十两"
number = re.compile(f"^((?P<bai>[{digit}]+)[百佰])?((?P<shi>[{digit}]*)[十拾])?(?P<ge>[{digit}]+)?$")


def str2number(num_str):
    if num_str in char2digit:
        return char2digit[num_str]

    match = number.match(num_str)
    if not match:
        return None

    return char2digit.get(match.group("bai"), 0) * 100 + char2digit.get(match.group("shi"), 0) * 10 + char2digit.get(match.group("ge"), 0)


serial_numbers = [r"(?P<number>[一二三四五六七八九十]+)、.+", r"(?P<number>[一二三四五六七八九十]+)\..+",
                  r"(?P<number>[一二三四五六七八九十]+)\s.+", r"(?P<number>\d+)\.[^\d]+",
                  r"\((?P<number>[一二三四五六七八九十]+)\).+", r"（(?P<number>[一二三四五六七八九十]+)）.+",
                  r"[(]?(?P<number>\d+)\)\.?.+", r"[(]?(?P<number>\d+)\)\s?.+",
                  r"[（]?(?P<number>\d+)）\.?.+", r"[（]?(?P<number>\d+)）\s?.+",
                  r"(?P<number>\d+)、.+", r"(?P<number>[①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳]).+"]

patterns = [re.compile(item) for item in serial_numbers]


def segment_paragraph(lines):
    paragraphs = []
    para = ""
    prev_len = None
    indent = None
    max_len = max(line.bbox[2] - line.bbox[0] for line in lines)
    max_len = min(460, max_len)
    for i, line in enumerate(lines):
        length = line.bbox[2] - line.bbox[0]
        text = line.get_text().strip()
        if not text:
            continue
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
        elif any(pattern.match(text) for pattern in patterns):
            if para:
                paragraphs.append(para)
            if length < max_len * 0.9:
                paragraphs.append(text)
                prev_len = None
                para = ""
            else:
                prev_len = length
                para = text
        elif prev_len is not None:
            if length < 0.9 * prev_len:
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
    pages = []
    top, bottom = extract_head_foot_note(pdf_file)
    logger.info("start to extract page elements")
    pages = extract_all_page_elements(pdf_file, top, bottom, table_settings)
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
                            prev_table.df = pd.concat([prev_table.df, element.df])
                            table = None
                if table:
                    all_elements.append(table)
    elements = arrange_elements(all_elements)

    return elements


def get_serial_number(match):
    number = match.group("number")
    serial_number = str2number(number)
    if serial_number is None:
        serial_number = int(number)

    return serial_number


def get_serial_number_from_element(element):
    line = element.get_text().strip()
    for pattern in patterns:
        match = pattern.match(line)
        if match:
            serial_number = get_serial_number(match)
            return serial_number

    return None


def is_valid_title(element, config):
    max_title_length = config["max_title_length"]
    if element.bbox[2] < max_title_length:
        return True
    if "exceptional_titles" not in config:
        return False
    for title in config["exceptional_titles"]:
        if title in element.get_text():
            return True
    return False


def segment_elements(elements, config={"max_title_length": 490}):
    stack = []
    current_serial_number = None
    target_indent = None
    noise_indent = None
    target_pattern = None
    parts = []
    sub_parts = []

    for element in elements:
        if not isinstance(element, LTTextLineHorizontal):
            sub_parts.append(element)
            continue
        matched = False
        line = element.get_text().strip()
        for pattern in patterns:
            match = pattern.match(line)
            if match:
                if not is_valid_title(element, config):
                    matched = False
                    break
                matched = True
                serial_number = get_serial_number(match)
                if target_pattern is None:
                    if serial_number == 1:
                        target_pattern = pattern
                        if sub_parts:
                            parts.append(sub_parts)
                        sub_parts = [element]
                        current_serial_number = 1
                        target_indent = element.bbox[0]
                    else:
                        matched = False
                elif pattern is not target_pattern:
                    matched = False
                else:
                    if serial_number == 1:
                        stack = [serial_number]
                        matched = False
                        noise_indent = element.bbox[0]
                    elif stack and serial_number - stack[-1] == 1:                           
                        if serial_number - current_serial_number == 1 and abs(element.bbox[0] - target_indent) < abs(element.bbox[0] - noise_indent):
                            stack = []
                            parts.append(sub_parts)
                            current_serial_number = serial_number
                            sub_parts = [element]
                            break
                        stack.append(serial_number)
                        matched = False
                    elif serial_number - current_serial_number == 1:
                        stack = []
                        parts.append(sub_parts)
                        current_serial_number = serial_number
                        sub_parts = [element]
                    else:
                        matched = False
                break

        if not matched:
            sub_parts.append(element)

    parts.append(sub_parts)

    return parts


def pdf2html(pdf_file, save_file, table_settings={}):
    elements = extract_pdf_elements(pdf_file, table_settings)
    parts = segment_elements(elements)

    with open("pdf.html", encoding="utf-8") as fi:
        html = fi.read()

    content = ""
    for part in parts:
        content += "=====================<br/>"
        for element in part:
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


def seperate_by_seperators(elements, seperators):
    parts = []
    sub_parts = []
    for element in elements:
        if not isinstance(element, LTTextLineHorizontal):
            sub_parts.append(element)
            continue
        text = element.get_text().strip()
        for seperator in seperators:
            if seperator in text:
                parts.append(sub_parts)
                sub_parts = [element]
                break
        else:
            sub_parts.append(element)

    if sub_parts:
        parts.append(sub_parts)

    return parts


def parse_segment(segment, config=None):
    title = None
    if isinstance(segment[0], LTTextLineHorizontal):
        line = segment[0].get_text().strip()
        for pattern in patterns:
            if pattern.match(line):
                title = segment[0]
                break
    if not title:
        segments = segment_elements(segment, config)
        parts = []
        content = segments[0]
        parts.append({"title": None, "content": content, "parts": None})
        parts.extend([parse_segment(segment, config) for segment in segments[1:]])
        return {"title": None, "content": None, "parts": parts}

    if len(segment) == 1:
        return {"title": title, "content": None, "parts": None}

    for key in config.get("explicit_seperators", []):
        if key in title:
            segments = seperate_by_seperators(segment[1:], config["explicit_seperators"][key])
            parts = [parse_segment(segment, config) for segment in segments]
            return {"title": title, "content": None, "parts": parts}

    if get_serial_number_from_element(segment[0]) == 1:
        segments = segment_elements(segment, config)
        if len(segments) != 1:
            parts = [parse_segment(segment, config) for segment in segments]
            return {"title": None, "content": None, "parts": parts}

    segments = segment_elements(segment[1:], config)
    content = None
    if len(segments) == 1:
        content = segments[0]
        return {"title": title, "content": content, "parts": None}
    else:
        parts = [parse_segment(segment, config) for segment in segments]
        return {"title": title, "content": content,  "parts": parts}


def refine_document(document):
    if document["parts"] is None:
        return document

    parts = document["parts"]
    parts = [refine_document(part) for part in parts]
    if document["content"] is None:
        if all(only_title(part) for part in parts):
            content = [part["title"] for part in parts]
            document["content"] = content
            return document

        if document["title"] is None:
            if len(parts) == 1:
                document["title"] = parts[0]["title"]
                document["content"] = parts[0]["content"]
                document["parts"] = parts[0]["parts"]
            elif all(only_parts(part) for part in parts):
                new_parts = []
                for part in parts:
                    new_parts.extend(part["parts"])
                document["parts"] = new_parts
        else:
            if len(parts) == 1 and parts[0]["title"] is None:
                document["content"] = parts[0]["content"]
                document["parts"] = parts[0]["parts"]
            elif all(only_parts(part) for part in parts):
                new_parts = []
                for part in parts:
                    new_parts.extend(part["parts"])
                document["parts"] = new_parts

    return document


def only_parts(document):
    return document["content"] is None and document["title"] is None


def only_title(document):
    return document["content"] is None and not document["parts"]


def convert_document(document):
    if document["title"]:
        document["title"] = document["title"].get_text().strip()
    if document["content"]:
        document["content"] = "".join(item.get_text() for item in document["content"] if isinstance(item, LTTextLineHorizontal))
    if document["parts"]:
        for part in document["parts"]:
            convert_document(part)


if __name__ == "__main__":

    # section_titles = parse_toc("data/report3.pdf")
    # segment_by_section("data/report3.pdf", section_titles, "data/zhanghang")

    # ts = {"engine": "camelot", "line_scale": 40, "strp_text": "\n", "split_text": True}
    # summarize_pdf2html("data/zhanghang/第三章管理层讨论与分析.pdf", "data/s5.html", ts)

    # for file in os.listdir("data/yili"):
    #     print(file)
    #     save_file = file.replace(".pdf", ".html")
    #     pdf2html(f"./data/yili/{file}", f"data/{save_file}", ts)

    # elements = extract_pdf_elements("./data/yili/2.pdf", ts)
    # content = "\n".join([element["text"] for element in elements if "text" in element])

    # print(content)

    # with open("data.txt", encoding="utf-8") as fi:
    #     content = fi.read()

    # content = segment_content(content)

    # with open("segmented.txt", "w", encoding="utf-8") as fo:
    #     fo.write("\n=============\n".join(content))

    import json

    pdf_file = "data/yili/第三节管理层讨论与分析.pdf"
    ts = {"engine": "camelot"}
    top, bottom = extract_head_foot_note(pdf_file)
    logger.info("start to extract page elements")
    pages = extract_all_page_elements(pdf_file, top, bottom, ts)
    elements = []
    for page in pages:
        elements.extend(page)
    config = {"exceptional_titles": ["公司因不适用准则规定或国家秘密、商业秘密等特殊原因"], "max_title_length": 490}

    segments = segment_elements(elements, config)
    print(len(segments))
    for i, segment in enumerate(segments):
        with open(f"data/segment{i}.txt", "w", encoding="utf-8") as fo:
            config = { #"explicit_seperators": {"报告期内主要经营情况": ["经营计划执行情况", "主营业务分析"]},
                    "exceptional_titles": ["坚守“伊利即品质”信条，为消费者提供安全、健康、高品质的产品和服务"],
                    "max_title_length": 490}
            document = parse_segment(segment, config)
            refine_document(document)
            convert_document(document)
            json.dump(document, fo, ensure_ascii=False, indent=4)
