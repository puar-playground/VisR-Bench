from PIL import Image
from utils.gpt4o import request_gpt4o
import os
import numpy as np
import matplotlib.pyplot as plt
import json
import re
os.system('clear')

class DocExtract(object):

    def __init__(self, file_dir):
        input_meta = json.load(open(file_dir, 'r'))
        self.meta = self.process(input_meta)
        self.page_info = input_meta['pages']
        self.n_page = len(self.page_info)
        self.content = self.extract_content()
    
    def process(self, input_meta):
    
        page_info = input_meta['pages']
        element_meta = input_meta['elements']

        processed_meta = []
        for e in element_meta:

            if 'Page' in e.keys():
                e_page = e['Page']
            elif 'Pages' in e.keys():
                e_page = e['Pages']
            else:
                continue

            page_w, page_h = page_info[e_page]['width'], page_info[e_page]['height']

            # try to get text
            if 'Text' in e.keys():
                e_text = e['Text']
                if 'Lang' in e.keys():
                    e_lang = e['Lang']
                else:
                    e_lang = None
            else:
                e_text = None
                e_lang = None

            # try to get image
            if 'filePaths' in e.keys():
                e_fig = e['filePaths']
            else:
                e_fig = None

            # check if bbox is given
            bbox_found = False
            if 'attributes' in e.keys():
                if 'BBox' in e['attributes'].keys():
                    bbox_found = True
                    bbox = e['attributes']['BBox']
                    e_box = [bbox[0] / page_w, bbox[1] / page_h, 
                            bbox[2] / page_w, bbox[3] / page_h]
            
            if not bbox_found and 'CharBounds' in e.keys():
                
                char_box = np.array(e['CharBounds'])
                bbox = [np.min(char_box[:, 0]), np.min(char_box[:, 1]), np.max(char_box[:, 2]), np.max(char_box[:, 3])]
                e_box = [bbox[0] / page_w, bbox[1] / page_h, 
                            bbox[2] / page_w, bbox[3] / page_h]

            e_box = [1-e_box[1], e_box[0], 1-e_box[3], e_box[2]]

            processed_meta.append({'e_page': e_page, 'e_text': e_text, 'e_lang': e_lang, 'e_box': e_box, 'e_fig': e_fig, 'has_table': ('Table' in e['Path'])})

        return processed_meta
    
    def get_page(self, page_index, k: str = None):

        if k is None:
            page_selected = [x for x in self.meta if x['e_page'] == page_index]
        elif k in {'e_page', 'e_text', 'e_lang', 'e_box', 'e_fig'}:
            page_selected = [x[k] for x in self.meta if x['e_page'] == page_index]
        else:
            raise Exception(f'No key in element metadata called {k}')

        return page_selected
    

    def extract_content(self):
        
        content = {i: [] for i in range(self.n_page)}
        for e in self.meta:
            e_page = e['e_page']

            if e['e_text'] is not None:
                content[e_page].append({'type': 'text', 'text': e['e_text'], 'lang': e['e_lang'], 'has_table': e['has_table']})

            if e['e_fig'] is not None:
                content[e_page].append({'type': 'image', 'images': e['e_fig'], 'has_table': e['has_table']})

        return content
    

    def get_table_page(self):
        table_exsit = []
        for e_page in self.content.keys():
            page_content = self.content[e_page]
            for e in page_content:
                has_table = e['has_table']
                if has_table:
                    table_exsit.append(e_page)
                    break
        return table_exsit

    def draw_boxes(self, page_index):
        """
        Draws boxes of a page on a blank canvas.
        """

        page_boxes = self.get_page(page_index=1, k='e_box')
        page_texts = self.get_page(page_index=1, k='e_text')
        page_texts = [str(t)[:7] for t in page_texts]

        selected_info = self.page_info[page_index]
        W, H = int(selected_info['width']), int(selected_info['height'])

        # Create a blank white canvas
        canvas = np.ones((H, W, 3), dtype=np.uint8) * 255

        fig, ax = plt.subplots(figsize=(W / 100, H / 100))
        ax.imshow(canvas)
        
        # Set axis limits and remove ticks
        ax.set_xlim(0, W)
        ax.set_ylim(H, 0)  # Invert y-axis
        ax.set_xticks([])
        ax.set_yticks([])

        # Draw boxes
        for i, ((top, left, bottom, right), box_t) in enumerate(zip(page_boxes, page_texts)):
            # Convert relative coordinates to absolute integer pixel values
            x1, y1 = int(left * W), int(top * H)
            x2, y2 = int(right * W), int(bottom * H)
            width, height = x2 - x1, y2 - y1

            # Draw rectangle
            rect = plt.Rectangle((x1, y1), width, height, edgecolor='red', facecolor='none', linewidth=2)
            ax.add_patch(rect)
            
            # Add text index
            ax.text(x1, max(0, y1 - 5), box_t, color='blue', fontsize=12, verticalalignment='bottom')

        plt.show()


def split_markdown_content(markdown_dir, image_prefix):
    """
    Splits a markdown string into a list, preserving text, tables, and grouping adjacent figures together.

    Parameters:
        markdown_text (str): The markdown content as a string.

    Returns:
        list: A list where each element is either:
            - A text string (normal text)
            - A dictionary {'table': table_content} for tables
            - A dictionary {'figures': [image_path1, image_path2, ...]} for grouped adjacent images.
    """

    md_lines = open(markdown_dir, 'r')
    markdown_text = ''.join([line for line in md_lines])

    # Regular expressions
    image_pattern = r"(!\[.*?\]\((.*?)\))"  # Matches markdown images
    table_pattern = r"(<table>.*?</table>)"  # Matches full tables (non-greedy)

    # Split markdown by both images and tables while preserving them
    parts = re.split(f"({image_pattern}|{table_pattern})", markdown_text, flags=re.DOTALL)

    structured_content = []
    figure_group = []  # Temporary storage for adjacent images

    for part in parts:
        if not part or part.isspace():
            continue  # Skip empty parts

        # Check if it's a table
        if re.match(table_pattern, part, flags=re.DOTALL):
            # If there was a previous figure group, add it before adding the table
            if figure_group:
                structured_content.append({"type": 'figures', 'content': figure_group})
                figure_group = []  # Reset the group
            structured_content.append({"type": 'table', 'content': part.strip()})

        # Check if it's an image
        elif re.match(image_pattern, part):
            image_match = re.search(image_pattern, part)
            if image_match:
                image_path = image_match.group(2).strip()
                figure_group.append(os.path.join(image_prefix, image_path))

        # Otherwise, treat it as text
        else:
            text = part.strip()
            if text:
                # If there was a previous figure group, add it before adding text
                if figure_group:
                    structured_content.append({"type": 'figures', 'content': figure_group})
                    figure_group = []  # Reset the group
                structured_content.append({"type": 'text', 'content': text})

    # If any remaining figures exist at the end, add them
    if figure_group:
        structured_content.append({"type": 'figures', 'content': figure_group})

    return structured_content

if __name__ == "__main__":


    extract = DocExtract(f'./CCpdf/1003f3ffdb94f347c9f7846388e766e8/1003f3ffdb94f347c9f7846388e766e8.json')

    # print content as a list of "text" and [image]
    print(extract.content)

    # plot the layout
    extract.draw_boxes(0)