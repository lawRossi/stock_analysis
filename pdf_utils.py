import os
import re

import fitz
from loguru import logger
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer, LTTextLineHorizontal
import pdfplumber
from pypdf import PdfReader, PdfWriter


section_title = re.compile(r"(?P<section_name>第\s*(\d+|[一二三四五六七八九十]+)\s*节.+)")


def find_toc_page(pdf_file):
    pdf = pdfplumber.open(pdf_file)
    toc_pattern = re.compile(r"^目\s*录$")
    for i in range(10):
        page = pdf.pages[i]
        text = page.extract_text().strip()
        lines = text.split("\n")
        for line in lines[:5]:
            if toc_pattern.match(line.strip()):
                return i
    return None


def segment_by_section(pdf_file, save_dir=None):
    toc_idx = find_toc_page(pdf_file)
    if toc_idx is None:
        logger.warning("toc not found")
        return

    pdf = PdfReader(pdf_file)
    sections = {}
    start = None
    section = None
    for i in range(toc_idx+1, len(pdf.pages)):
        page = pdf.pages[i]
        for line in page.extract_text().strip().split("\n"):
            match = section_title.search(line.strip())
            if match:
                if start is not None:
                    sections[section.strip()] = (start, i)
                start = i
                section = match.group("section_name")
    if section:
        sections[section.strip()] = (start, len(pdf.pages)-1)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        for section, (start, end) in sections.items():
            writer = PdfWriter()
            for i in range(start, end+1):
                writer.add_page(pdf.pages[i])
            file_name = section.replace(" ", "")
            writer.write(f"{save_dir}/{file_name}.pdf")

    return sections


def extract_sections(report_file, section_titles, save_path):
    sections = segment_by_section(report_file)
    pages = set()
    for section, page_range in sections.items():
        if any(title in section for title in section_titles):
            for page in range(page_range[0], page_range[1]+1):
                pages.add(page)

    pdf = PdfReader(report_file)
    pdf_writer = PdfWriter()
    for page in sorted(pages):
        pdf_writer.add_page(pdf.pages[page])

    with open(save_path, "wb") as fo:
        pdf_writer.write(fo)


def get_sorted_lines(page):
    lines = []
    for element in page:
        if isinstance(element, LTTextContainer):
            for text_line in element:
                if isinstance(text_line, LTTextLineHorizontal):
                    lines.append((text_line, text_line.bbox[3]))
    lines = [item[0] for item in sorted(lines, key=lambda x: x[1], reverse=True)]

    return lines


def extract_head_foot_note(pdf_file, head_lines=1, foot_lines=1):
    pages = extract_pages(pdf_file)
    page = next(pages)
    lines = get_sorted_lines(page)
    width, height = page.bbox[2], page.bbox[3]
    top = lines[head_lines-1].bbox[1]
    bottom = lines[-foot_lines].bbox[-1]
    return [(0, 0, width, bottom), (0, top, width, height)]


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


def extract_tables(pdf_file, page_idx):
    pdf = pdfplumber.open(pdf_file)
    page = pdf.pages[page_idx]
    tables = page.find_tables()

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


def extract_page_elements(pdf_file, page_idx):
    pages = extract_pages(pdf_file, page_numbers=[page_idx])
    page = next(pages)
    height = page.bbox[3]
    tables = extract_tables(pdf_file, page_idx)
    images = extract_images(pdf_file, page_idx)
    elements = [(table, convert_bbox(table.bbox, height)) for table in tables] + [(image, convert_bbox(image["bbox"], height)) for image in images]
    bboxes = extract_head_foot_note(pdf_file)
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


serial_numbers = [r"(?P<number>[一二三四五六七八九十]+)([、.]|\s).+", r"(?P<number>\d+)\.[^\d]+", r"\((?P<number>[一二三四五六七八九十]+)\).+",
                  r"\((?P<number>\d+)\)[.|\s].+", r"(?P<number>\d+)、.+"]
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
    for i, line in enumerate(lines):
        length = line.bbox[2] - line.bbox[0]
        text = line.get_text().strip()
        if i > 0:
            indent = line.bbox[0] - lines[i-1].bbox[0]

        if indent is not None and indent > 20:
            if para:
                paragraphs.append(para)
            para = text
            prev_len = length
        elif prev_len is not None:
            if any(pattern.match(text) for pattern in patterns) or prev_len < length:
                paragraphs.append(para)
                paragraphs.append(text)
                prev_len = None
                para = ""
            elif prev_len - length > prev_len * 0.1:
                para += text
                paragraphs.append(para)
                para = ""
                prev_len = None
            else:
                para += text
                prev_len = length
        else:
            para = text
            prev_len = length

    if para:
        paragraphs.append(para)

    return paragraphs


def arrange_elements(elements):
    lines = [element["line"] for element in elements if "line" in element]
    paragraphs = segment_paragraph(lines)
    i = -1
    new_elements = []
    for element in elements:
        if "line" in element:
            if i == -1:
                i += 1
                new_elements.append({"text": paragraphs[i]})
            text = element["line"].get_text().strip()
            if text not in paragraphs[i]:
                i += 1
                assert text in paragraphs[i]
                new_elements.append({"text": paragraphs[i]})
        else:
            new_elements.append(element)

    return new_elements


def extract_section_elements(pdf_file):
    pdf = pdfplumber.open(pdf_file)
    page_num = len(pdf.pages)
    pdf.close()
    pages = []
    for i in range(page_num):
        elements = extract_page_elements(pdf_file, i)
        pages.append(elements)
    if len(pages) > 1:
        for i, element in enumerate(pages[-1]):
            if isinstance(element, LTTextLineHorizontal):
                if section_title.search(element.get_text()):
                    pages[-1] = pages[-1][:i]
                    if not pages[-1]:
                        pages.pop()
                    break
    all_elements = []
    for page in pages:
        for element in page:
            if isinstance(element, LTTextLineHorizontal):
                all_elements.append({"line": element})
            elif isinstance(element, dict):
                all_elements.append(element)
            else:
                if all_elements and "line" in all_elements[-1]:
                    table = {"table": element}
                    text = all_elements[-1]["line"].get_text()
                    if "单位:" in text or "单位：" in text:
                        head_note = text.strip()
                        all_elements.pop()
                        table["head_note"] = head_note
                    all_elements.append(table)
    elements = arrange_elements(all_elements)

    return elements


if __name__ == "__main__":
    from api import download_annual_reports

    # download_annual_reports("格力电器", "data", 1)
    # segment_by_section("./data/格力电器2022年年度报告.pdf", "data")

    elements = extract_section_elements("./data/第二节公司简介和主要财务指标.pdf")
    lines = []
    for element in elements:
        if "text" in element:
            print(element["text"])
        elif "image" in element:
            print("image")
        else:
            print("table")

    # paras = segment_paragraph(lines)
    # for para in paras:
    #     print(para + "\n\n")

    # pdf = PdfReader("./data/格力电器2022年年度报告.pdf")
    # print(pdf.pages[5].extract_text())
