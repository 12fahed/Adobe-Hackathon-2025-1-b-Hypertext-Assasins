import difflib
import fitz  # PyMuPDF
import pandas as pd
import re
from collections import Counter
import json
from titlecase import titlecase
from sklearn.cluster import KMeans
import numpy as np
import textstat
from langdetect import detect, LangDetectException
import os

BULLET_UNICODE_SET = {
    '\u2022',  # • Bullet
    '\u2023',  # ‣ Triangular Bullet
    '\u25E6',  # ◦ White Bullet
    '\u2219',  # ∙ Bullet Operator
    '\u00B7',  # · Middle Dot
    '\u2043',  # ⁃ Hyphen Bullet
    '\u2013',  # – En Dash
    '\u25AA',  # ▪ Black Small Square
    '\u25AB',  # ▫ White Small Square
    '\u25A0',  # ■ Black Square
    '\u25A1',  # □ White Square
    '\u27A4',  # ➤ Black Rightwards Arrowhead
    '\u2794',  # ➔ Heavy Wide-Headed Rightwards Arrow
    '\u2726',  # ✦ Black Four-Pointed Star
    '\u2727',  # ✧ White Four-Pointed Star
     "-", "+", "•", "‣", "▪", "∙", "■", "●", "○"  # ASCII/visual fallback
}

def hex_to_rgb(color_hex):
    """Convert PyMuPDF color int to RGB tuple"""
    r = (color_hex >> 16) & 0xff
    g = (color_hex >> 8) & 0xff
    b = color_hex & 0xff
    return (r, g, b)

def calculate_indentation(block, page_width):
    """Returns indentation level (0-1 ratio)"""
    return block["bbox"][0] / page_width 


def detect_columns(blocks, n_columns=2):
    """Cluster blocks into columns using x-coordinates"""
    if not blocks:
        return []
    
    # Correct array construction
    valid_blocks = [b for b in blocks if "bbox" in b]
    if not valid_blocks:
        return [0] * len(blocks)
    
    x_coords = np.array([b["bbox"][0]] for b in valid_blocks)
    
    if len(x_coords) < n_columns:
        return [0] * len(blocks)
    
    try:
        kmeans = KMeans(n_clusters=min(n_columns, len(x_coords))).fit(x_coords)
        # Map back to original blocks
        labels = []
        valid_idx = 0
        for b in blocks:
            if "bbox" in b:
                labels.append(int(kmeans.labels_[valid_idx]))
                valid_idx += 1
            else:
                labels.append(0)
        return labels
    except Exception as e:
        print(f"Column detection failed: {str(e)}")
        return [0] * len(blocks)

def is_header_footer(block, page_height):
    """Check if block is in top/bottom 10% of page"""
    top_y = block["bbox"][1] / page_height
    bottom_y = block["bbox"][3] / page_height
    return {
        "is_header": top_y < 0.1,  # Top 10%
        "is_footer": bottom_y > 0.9  # Bottom 10%
    }

def get_readability(text):
    return {
        "flesch_reading_ease": textstat.flesch_reading_ease(text),
        "smog_index": textstat.smog_index(text),
        "coleman_liau": textstat.coleman_liau_index(text)
    }

def detect_language(text):
    try:
        return detect(text[:500])  # Sample first 500 chars for speed
    except LangDetectException:
        return "unknown"

def extract_hyperlinks(page):
    """Extract all links with their text contexts"""
    links = []
    for link in page.get_links():
        if link["kind"] == 1:  # URI links only
            links.append({
                "uri": link.get("uri", ""),
                "rect": link["from"],
                "link_text": page.get_text("text", clip=link["from"])
            })
    return links

def is_title_case(text):
    return text.strip() == titlecase(text.strip())

def starts_with_bullet(text):
    text = text.strip()
    return any(text.startswith(bullet) for bullet in BULLET_UNICODE_SET)


def merge_multiline_headers(blocks, max_line_spacing=6, font_size_threshold=1.5):
    merged_blocks = []
    temp_block = None

    for block in blocks:
        if not temp_block:
            temp_block = block.copy()
            continue

        # Don't merge if pages are different
        if block["page"] != temp_block["page"]:
            merged_blocks.append(temp_block)
            temp_block = block.copy()
            continue

        same_style = (
            abs(block["avg_font_size"] - temp_block["avg_font_size"]) < font_size_threshold and
            block["is_bold"] == temp_block["is_bold"] and
            block["is_italic"] == temp_block["is_italic"]
        )

        close_vertically = block["spacing_above"] < max_line_spacing

        if same_style and close_vertically:
            temp_block["text"] += " " + block["text"]
            temp_block["spacing_above"] = min(temp_block["spacing_above"], block["spacing_above"])
        else:
            merged_blocks.append(temp_block)
            temp_block = block.copy()

    if temp_block:
        merged_blocks.append(temp_block)

    return merged_blocks

def detect_font_styles(font_name, span_flags=None):
    """
    Classify font into categories based on name patterns and flags.
    Returns: dict with font style flags
    """
    font_name = font_name.lower()
    styles = {
        # Typeface classification
        'is_serif': any(x in font_name for x in ['times', 'cambria', 'garamond', 'georgia', 'serif', 'bookman']),
        'is_sans_serif': any(x in font_name for x in ['arial', 'helvetica', 'calibri', 'verdana', 'sans', 'roboto', 'futura']),
        'is_monospace': any(x in font_name for x in ['courier', 'consolas', 'mono', 'fixed', 'source code']),
        'is_script': any(x in font_name for x in ['script', 'hand', 'comic', 'cursive', 'brush', 'calligraph']),
        'is_symbol': any(x in font_name for x in ['symbol', 'wingdings', 'dingbat', 'zapf']),
        'is_display': any(x in font_name for x in ['impact', 'broadway', 'jokerman', 'stencil', 'cooper']),
        
        # Weight
        'is_light': any(x in font_name for x in ['light', 'thin', 'hair', 'extra light']),
        'is_regular': 'regular' in font_name or not any(x in font_name for x in ['bold', 'light', 'black']),
        'is_bold': 'bold' in font_name or 'heavy' in font_name or (span_flags and span_flags & 2**4),
        'is_black': any(x in font_name for x in ['black', 'ultra', 'extra bold']),
        
        # Slant
        'is_italic': 'italic' in font_name or (span_flags and span_flags & 2**1),
        'is_oblique': 'oblique' in font_name,
        
        # Width
        'is_condensed': any(x in font_name for x in ['condensed', 'narrow', 'compressed']),
        'is_expanded': any(x in font_name for x in ['expanded', 'extended', 'wide']),
    }
    
    # Ensure mutual exclusivity for some categories
    if styles['is_light']:
        styles['is_regular'] = False
    if styles['is_black']:
        styles['is_bold'] = False
    if styles['is_oblique']:
        styles['is_italic'] = False
        
    return styles

def extract_pdf_features_with_merging_bullet_spacing(pdf_path):
    doc = fitz.open(pdf_path)
    raw_blocks = []
    font_sizes = []
    font_counts = Counter()

    # First pass: gather font size stats
    for page in doc:
        for b in page.get_text("dict")["blocks"]:
            if "lines" not in b:
                continue
            for line in b["lines"]:
                for span in line["spans"]:
                    font_sizes.append(span["size"])
                    font_counts[span["font"]] += 1

    median_font_size = pd.Series(font_sizes).median()
    dominant_font = font_counts.most_common(1)[0][0]

    # Second pass: extract raw blocks with spatial info
    for page_num, page in enumerate(doc):
        page_height = page.rect.height
        page_width = page.rect.width

        hyperlinks = []
        for link in page.get_links():
            if link["kind"] == 1:  # URI links
                try:
                    link_text = page.get_text("text", clip=link["from"])[:200]
                    hyperlinks.append({
                        "uri": link.get("uri", ""),
                        "rect": link["from"],
                        "link_text": link_text
                    })
                except:
                    continue

        blocks = [b for b in page.get_text("dict")["blocks"] if "lines" in b]
        sorted_blocks = sorted(blocks, key=lambda b: b["bbox"][1])

        prev_bottom = 0
        # for b in blocks:
        for b in sorted_blocks:
            text = ""
            fonts, sizes, y_positions = [], [], [],
            span_styles = []

            for line in b["lines"]:
                for span in line["spans"]:
                    text += span["text"] + " "
                    fonts.append(span["font"])
                    sizes.append(span["size"])
                    y_positions.append(span["bbox"][1])
                    span_styles.append(detect_font_styles(span["font"], span["flags"]))

            if not text.strip():
                continue
        
             # Combine styles from all spans in the block
            combined_styles = {
                'is_bold': any(s['is_bold'] for s in span_styles),
                'is_italic': any(s['is_italic'] for s in span_styles),
                'is_serif': any(s['is_serif'] for s in span_styles),
                'is_sans_serif': any(s['is_sans_serif'] for s in span_styles),
                'is_monospace': any(s['is_monospace'] for s in span_styles),
                'is_script': any(s['is_script'] for s in span_styles),
                'is_symbol': any(s['is_symbol'] for s in span_styles),
                'is_display': any(s['is_display'] for s in span_styles),
                'is_light': any(s['is_light'] for s in span_styles),
                'is_black': any(s['is_black'] for s in span_styles),
                'is_condensed': any(s['is_condensed'] for s in span_styles),
                'is_expanded': any(s['is_expanded'] for s in span_styles),
            }

            avg_font_size = sum(sizes) / len(sizes)
            font_name = fonts[0] if fonts else dominant_font

            top_y, bottom_y = b["bbox"][1], b["bbox"][3]
            spacing_above = top_y - prev_bottom if prev_bottom > 0 else 0
            prev_bottom = bottom_y
            color_rgb = hex_to_rgb(span["color"])
            is_underlined = bool(span["flags"] & 2**3)
            is_strikethrough = bool(span["flags"] & 2**5)
            hf_flags = is_header_footer(b, page_height)
            readability = get_readability(text.strip())

            raw_blocks.append({
                "page": page_num + 1,
                "text": text.strip(),
                "language": detect_language(text.strip()),
                "avg_font_size": avg_font_size,
                "font_size_diff": avg_font_size - median_font_size,
                "font_diff_from_body": font_name != dominant_font,
                "font_name": font_name,
                **combined_styles,
                "text_color_r": color_rgb[0],
                "text_color_g": color_rgb[1],
                "text_color_b": color_rgb[2],
                "is_black_text": all(c < 50 for c in color_rgb),
                "is_underlined": is_underlined,
                "indentation_ratio": calculate_indentation(b, page_width),
                "is_strikethrough": any(
                    bool(span["flags"] & 2**5) 
                    for line in b.get("lines", []) 
                    for span in line.get("spans", [])
                ),
                **hf_flags,
                **readability,
                "caps_ratio": sum(1 for c in text if c.isupper()) / max(len(text), 1),
                "is_title_case": is_title_case(text),
                "is_short": len(text.split()) < 10,
                "starts_with_numbering": bool(re.match(r"^(\d+[\.\)]|[A-Z][\.\)]|\(?[ivx]+\)?)[\s\-:]", text.strip(), re.IGNORECASE)),
                "is_center_aligned": abs(b["bbox"][0] - (page_width - b["bbox"][2])) < 100,
                "y_position_relative": top_y / page_height,
                "bottom_y_relative": bottom_y / page_height,
                "spacing_above": spacing_above,
                "page_height": page_height,
                "is_bullet_point": starts_with_bullet(text),
            })

        if hyperlinks:
            raw_blocks.append({
                "page": page_num + 1,
                "is_hyperlink_collection": True,
                "hyperlinks": hyperlinks
            })

    if raw_blocks:
        try:
            column_labels = detect_columns([b for b in raw_blocks if not b.get("is_hyperlink_collection", False)])
            for i, block in enumerate(raw_blocks):
                if not block.get("is_hyperlink_collection", False):
                    block["column_group"] = int(column_labels[i]) if i < len(column_labels) else 0
        except Exception as e:
            print(f"Column detection failed: {str(e)}")
            for block in raw_blocks:
                block["column_group"] = 0

    # Merge adjacent header lines
    merged_blocks = merge_multiline_headers(raw_blocks)

    # Clean result
    for block in merged_blocks:
        block.pop("bottom_y_relative", None)
        block.pop("page_height", None)

    return pd.DataFrame(merged_blocks)

def extract_possible_toc_entries(text_lines):
    """
    Heuristic to extract TOC-like entries: lines with section titles and page numbers
    Example pattern: "1 Introduction ........ 1"
    """
    toc_entries = []
    toc_pattern = re.compile(r"(.+?)\s+\.{2,}\s+(\d+)$")
    for line in text_lines:
        match = toc_pattern.match(line.strip())
        if match:
            toc_text = match.group(1).strip()
            toc_entries.append(toc_text)
    return toc_entries

def find_toc_entries(doc, max_pages=3):
    """
    Extract TOC entries from the first few pages.
    """
    toc_lines = []
    for i in range(min(max_pages, len(doc))):
        page_text = doc[i].get_text("text")
        toc_lines.extend(page_text.splitlines())
    return extract_possible_toc_entries(toc_lines)

def add_toc_flags_to_blocks(blocks_df, doc, similarity_threshold=0.85):
    """
    Add in_TOC_flag to each block if it matches a TOC entry.
    """
    toc_entries = find_toc_entries(doc)
    toc_flags = []
    for _, row in blocks_df.iterrows():
        block_text = row["text"].strip()
        # Compute best fuzzy match against TOC entries
        match = difflib.get_close_matches(block_text, toc_entries, n=1, cutoff=similarity_threshold)
        toc_flags.append(bool(match))
    blocks_df["in_TOC_flag"] = toc_flags
    return blocks_df

def extract_pdf_features_full_with_bullets(pdf_path):
    """
    Full pipeline: extract features, merge multi-line headers, match with TOC.
    """
    doc = fitz.open(pdf_path)
    df = extract_pdf_features_with_merging_bullet_spacing(pdf_path)
    df = add_toc_flags_to_blocks(df, doc)
    return df

os.makedirs('./app/extracted-data', exist_ok=True)

failedList = []

# Process all PDFs in the data folder
for filename in os.listdir('./app/input'):
    if filename.lower().endswith('.pdf'):
        pdf_path = os.path.join('./app/input', filename)
        
        try:
            print(f"Processing {filename}...")
            
            # Extract features
            df = extract_pdf_features_full_with_bullets(pdf_path)
            
            # Convert to JSON
            json_output = df.to_json(orient='records', indent=2)
            
            # Create output filename
            json_filename = os.path.splitext(filename)[0] + '.json'
            json_path = os.path.join('./app/extracted-data', json_filename)
            
            # Save JSON
            with open(json_path, 'w') as f:
                f.write(json_output)
                
            print(f"Successfully processed {filename} -> {json_filename}")
            
        except Exception as e:
            print(f"Failed to process {filename}: {str(e)}")
            failedList.append(filename)
            print(failedList)

print("All PDF processing complete!")
print(f"Failed List: {failedList}")