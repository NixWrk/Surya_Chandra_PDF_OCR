import json

from lxml import etree


def create_hocr_document(target_page_bbox, lines):
    # Create the root <html> element with namespaces and language attributes
    XML_NS = "http://www.w3.org/XML/1998/namespace"

    # Create the <html> element with attributes
    html = etree.Element(
        "html",
        attrib={
            "xmlns": "http://www.w3.org/1999/xhtml",
            "{%s}lang" % XML_NS: "en",  # xml:lang attribute with the correct namespace
            "lang": "en",  # lang attribute
        },
    )

    # Create the <head> element
    head = etree.SubElement(html, "head")

    # Add the <title> element inside <head>
    etree.SubElement(head, "title")

    # Add the <meta> elements inside <head>
    etree.SubElement(
        head,
        "meta",
        attrib={"http-equiv": "Content-Type", "content": "text/html;charset=utf-8"},
    )
    etree.SubElement(
        head, "meta", attrib={"name": "ocr-system", "content": "surya ocr"}
    )
    etree.SubElement(
        head,
        "meta",
        attrib={
            "name": "ocr-capabilities",
            "content": "ocr_page ocr_carea ocr_par ocr_line ocrx_word ocrp_wconf",
        },
    )

    # Create the <body> element
    body = etree.SubElement(html, "body")

    # Add the <div> element inside <body> with specified attributes
    ocr_page = etree.SubElement(
        body,
        "div",
        attrib={
            "class": "ocr_page",
            "title": f'image "output2/000001_ocr.png"; bbox {target_page_bbox[0]} {target_page_bbox[1]} {target_page_bbox[2]} {target_page_bbox[3]}; ppageno 0; scan_res 300 300',
        },
    )

    for line in lines:
        ocr_page.append(line)

    # Create a DOCTYPE for XHTML 1.0 Transitional
    doctype = '<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Transitional//EN" "http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd">'

    xhtml_content = etree.tostring(
        html, pretty_print=True, encoding="utf-8", doctype=doctype, xml_declaration=True
    ).decode("utf-8")

    return xhtml_content


def create_hocr_line(bbox: list, text: str, confidence: float) -> str:
    """Create a part of the HOCR structure for a given text line."""

    bbox_str = f"{int(bbox[0])} {int(bbox[1])} {int(bbox[2])} {int(bbox[3])}"
    # Create the root element <div>
    div_ocr_carea = etree.Element(
        "div",
        attrib={
            "title": f"bbox {bbox_str}",
            "class": "ocr_carea",
        },
    )

    # Create the <p> element
    p = etree.SubElement(
        div_ocr_carea, "p", attrib={"title": f"bbox {bbox_str}", "class": "ocr_par"}
    )

    # Create the <span> element for ocr_line
    line = etree.SubElement(
        p, "span", attrib={"title": f"bbox {bbox_str}", "class": "ocr_line"}
    )

    # Create the <span> element for ocrx_word
    word = etree.SubElement(
        line,
        "span",
        attrib={"title": f"bbox {bbox_str}; x_wconf {int(confidence*100)}", "class": "ocrx_word"},
    )

    # Add the text content for the word
    word.text = text

    return div_ocr_carea


def convert_surya_result_to_hocr(
    surya_result_file, surya_doc_name, target_bbox, output_file
):
    with open(surya_result_file) as f:
        data = json.load(f)

    page0 = data[surya_doc_name][0]
    source_bbox = page0["image_bbox"]
    bbox_scale_x = target_bbox[2] / source_bbox[2]
    bboy_scale_y = target_bbox[3] / source_bbox[3]

    converted_lines_as_etree_elements = []
    for text_line in page0["text_lines"]:
        # Calculate the scaled bounding box
        scaled_bbox = [
            text_line["bbox"][0] * bbox_scale_x,
            text_line["bbox"][1] * bboy_scale_y,
            text_line["bbox"][2] * bbox_scale_x,
            text_line["bbox"][3] * bboy_scale_y,
        ]
        # Create the HOCR line
        line = create_hocr_line(
            bbox=scaled_bbox, text=text_line["text"], confidence=text_line["confidence"]
        )
        converted_lines_as_etree_elements.append(line)

    text_docu = create_hocr_document(target_bbox, converted_lines_as_etree_elements)

    with open(output_file, "w") as f:
        f.write(text_docu)
    print(f"File {output_file} has been written successfully.")

def main():
    surya_result_file = "results/surya/docu1test/results.json"
    surya_doc_name = "docu1test"
    target_bbox = [0, 0, 2480, 3360]
    output_file = "ocrmypdf_output/000001_ocr_hocr.hocr"
    convert_surya_result_to_hocr(surya_result_file, surya_doc_name, target_bbox, output_file)
    

if __name__ == "__main__":
    main()
