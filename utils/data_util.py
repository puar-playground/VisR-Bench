from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
import json
import re
os.system('clear')


class QAData(object):

    def __init__(self, QA_file_dir: str, data_root_dir: str='.', multilingual=False):

        self.data_root_dir = data_root_dir
        self.qa_file = QA_file_dir
        self.multilingual = multilingual
        self.meta_list = json.load(open(self.qa_file, 'r'))
        self.doc_names = self.get_doc_list()

        # self.flat_qa_list = self.qa_serialize()
        self.doc_qa_list = self.qa_doc_batch()

        self.n_qa = sum([len(x['qa_list']) for x in self.doc_qa_list])
        self.n_doc = len(self.doc_names)
        self.n_page = len(self.meta_list)

        print(f' n_doc: {self.n_doc}, n_page: {self.n_page}, n_qa: {self.n_qa}')
        

        # markdown_meta = self.split_markdown_content(os.path.join(QA_file_dir), image_prefix=)

    def get_doc_list(self):
        name = set([x['file_name'] for x in self.meta_list])
        return name    

    def qa_serialize(self):
        
        format_out = []
        for page_meta in self.meta_list:
            file_name = page_meta['file_name']
            page_index = page_meta['page']
            page_image_dir = os.path.join(self.data_root_dir, f'{file_name}/images', file_name + f'_{page_index-1}.png')
            all_page_image_dir = self.sort_page(os.listdir(os.path.join(self.data_root_dir, f'{file_name}/images')), pattern=r'_(\d+)\.png$', extension='.png')
            all_page_image_dir = [os.path.join(self.data_root_dir, f'{file_name}/images', str(x)) for x in all_page_image_dir]

            # all_page_md_dir = self.sort_page(os.listdir(os.path.join(self.data_root_dir, f'{file_name}/')), pattern=r'_(\d+)\.md$', extension='.md')
            all_page_md_dir = [os.path.join(self.data_root_dir, f'{file_name}', f'{file_name}_{x}.md') for x in range(len(all_page_image_dir))]

            if len(all_page_image_dir) < 2:
                continue

            try:
                if not self.multilingual and page_meta['QA']['detected_language'] != 'English':
                    continue

                for qa in page_meta['QA']['questions']:
                    question = qa['question_in_detected_language']
                    if 'answer_in_detected_language' not in qa.keys():
                        answer = ''
                    else:
                        answer = qa['answer_in_detected_language']

                    format_out.append({'page_image_dir': page_image_dir, 'question': question, 'answer': answer, 
                                        page_index: page_index-1, 'all_page_image_dir': all_page_image_dir, 
                                        'all_page_md_dir': all_page_md_dir})
            except:
                continue

        return format_out
    
    def qa_doc_batch(self):

        format_dict = {k: {'qa_list': [], 'all_page_image_dir': []} for k in self.doc_names}
        for page_meta in self.meta_list:
            file_name = page_meta['file_name']
            page_index = page_meta['page']
            all_page_image_dir = self.sort_page(os.listdir(os.path.join(self.data_root_dir, f'{file_name}/images')), pattern=r'_(\d+)\.png$', extension='.png')
            all_page_image_dir = [os.path.join(self.data_root_dir, f'{file_name}/images', str(x)) for x in all_page_image_dir]

            all_page_md_dir = self.sort_page(os.listdir(os.path.join(self.data_root_dir, f'{file_name}')), pattern=r'_(\d+)\.md$', extension='.md')
            # all_page_md_dir = [os.path.join(self.data_root_dir, f'{file_name}/', x) for x in all_page_md_dir]
            # all_page_md_str = [self.load_md(x) for x in all_page_md_dir]

            all_page_md_dir = [os.path.join(self.data_root_dir, f'{file_name}', f'{file_name}_{x}.md') for x in range(len(all_page_image_dir))]
            all_page_md_str = [self.load_md(x) if os.path.isfile(x) else 'None' for x in all_page_md_dir]
            
            if len(all_page_image_dir) < 2:
                continue
            
            format_dict[file_name]['all_page_image_dir'] = all_page_image_dir.copy()
            format_dict[file_name]['all_page_md_str'] = all_page_md_str.copy()

            format_dict[file_name]['file_name'] = file_name

            try:
                if not self.multilingual and page_meta['QA']['detected_language'] != 'English':
                    continue
                for qa in page_meta['QA']['questions']:
                    question = qa['question_in_detected_language']
                    if 'answer_in_detected_language' not in qa.keys():
                        answer = ''
                    else:
                        answer = qa['answer_in_detected_language']

                    format_dict[file_name]['qa_list'].append({'question': question, 'answer': answer, 
                                                              'page_index': page_index-1, 
                                                              'detected_language': page_meta['QA']['detected_language']})
            except:
                continue

        format_out = sorted([x for x in format_dict.values() if len(x['qa_list']) > 0], key=lambda x: x['file_name'])

        return format_out


    @staticmethod
    def sort_page(filenames, pattern, extension=None):
        """
        Filters and sorts a list of filenames based on a custom regex pattern to extract the page number.
        
        Args:
            filenames (list of str): List of filenames to be filtered and sorted.
            pattern (str): Regular expression pattern with a capturing group for the page number.
            extension (str, optional): File extension to filter (e.g., '.png'). If None, all files are considered.
        
        Returns:
            list of str: Sorted filenames.
        """
        # Filter filenames based on the extension if provided
        if extension:
            filenames = [f for f in filenames if f.lower().endswith(extension.lower())]

        def extract_page_number(filename):
            match = re.search(pattern, filename)
            return int(match.group(1)) if match else float('inf')  # Assigns a high value if no match

        return sorted(filenames, key=extract_page_number)

    @staticmethod
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

    @staticmethod
    def load_md(md_dir):
        md_file = open(md_dir, 'r')
        md_str = ''.join([line for line in md_file])
        return md_str

if __name__ == "__main__":

    d = QAData(QA_file_dir='/Users/chenjian/Projects/M4Doc/M4Doc_local/Local_QA/text_QA.json', 
               data_root_dir='/Users/chenjian/Projects/M4Doc/M4Doc_local')
    
    # print(d.doc_qa_list[10])