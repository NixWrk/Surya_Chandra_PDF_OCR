import os

import re
import sys
from pathlib import Path

import ocrmypdf

import converter_surya_hocr

from lxml import etree

from ocr_text import main as ocr_text_main


def file_exists(file_path):
    return os.path.isfile(file_path)


def move_file(input_file, output_file):
    if not os.path.isfile(input_file):
        print(f"file {input_file} does not exist")
        return
    os.rename(input_file, output_file)
    print(f"File {input_file} has been moved to {output_file}.")


def get_bbox(input_file):
    with open(input_file, "r") as f:
        tree = etree.parse(f)

        # {'class': 'ocr_page', 'id': 'page_1', 'title': 'image "my_docu/000001_ocr.png"; bbox 0 0 2480 3360; ppageno 0; scan_res 300 300'}
        for element in tree.iter():
            att = element.attrib
            if att.has_key("class"):
                if att["class"] == "ocr_page":
                    title = att["title"]
                    bbox = extract_bbox_from_title(title)
                    return bbox


def extract_bbox_from_title(title):
    regex = r"bbox\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)"
    match = re.search(regex, title)
    if match:
        x1, y1, x2, y2 = map(int, match.groups())
        return (x1, y1, x2, y2)
    else:
        return None


if __name__ == "__main__":

    # Generate file paths
    input_doc_name = "docu1test"
    input_filename = f"{input_doc_name}.pdf"
    surya_output_path = f"./results/surya/{input_doc_name}"
    surya_result = f"{surya_output_path}/results.json"
    ocrmypdf_output_path = "./ocrmypdf_output/"
    ocrmypdf_page1_file = f"{ocrmypdf_output_path}000001_ocr_hocr.hocr"
    ocrmypdf_page1_backup = f"{ocrmypdf_output_path}000001_ocr_hocr.hocr.backup"
    doc_name_output = f"{input_doc_name}_ocrmypdf_with_surya_ocr.pdf"

    print("Making sure ocrmypdf_output folder: {ocrmypdf_output_path} exists ...")
    os.makedirs(ocrmypdf_output_path, exist_ok=True)

    print(f"Converting inputpdf {input_filename} to hocr using ocrmypdf api...")
    ocrmypdf.api._pdf_to_hocr(
        input_pdf=Path(input_filename), output_folder=Path(ocrmypdf_output_path)
    )

    print("Extracting bounding box of page from hocr file...")
    hocr_page_bbox = get_bbox(ocrmypdf_page1_file)
    print(hocr_page_bbox)

    if not file_exists(ocrmypdf_page1_backup):
        print(f"Make backup of original hocr file from ocrmypdf as {ocrmypdf_page1_backup}...")
        # Move to backup file if it does not exist
        move_file(ocrmypdf_page1_file, ocrmypdf_page1_backup)

    print(f"Run surya ocr on the {input_filename} to get surya json-results...")
    sys.argv[0] = re.sub(r"(-script\.pyw|\.exe)?$", "", sys.argv[0])
    sys.argv.append(input_filename)
    sys.argv.append("--langs")
    sys.argv.append("de")
    ocr_text_main()

    print("Converting surya results to hocr. Replacing hocr file from ocrmypdf...")
    converter_surya_hocr.convert_surya_result_to_hocr(
        surya_result, input_doc_name, hocr_page_bbox, ocrmypdf_page1_file
    )

    print("Converting hocr to pdf using ocrmypdf api...")
    ocrmypdf.api._hocr_to_ocr_pdf(
        work_folder=Path(ocrmypdf_output_path), output_file=Path(doc_name_output)
    )

    print(f"Generated output pdf: {doc_name_output}")
