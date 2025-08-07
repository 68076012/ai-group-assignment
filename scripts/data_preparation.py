# scripts/data_preparation.py

import os
# from datasets import Dataset
import json
import re


def load_stories_from_folder(folder_path):
    """
    โหลดเนื้อหานิทานจากไฟล์ .txt ทั้งหมดในโฟลเดอร์
    """
    stories = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            filepath = os.path.join(folder_path, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                stories.append(f.read())
    return stories


def load_data_from_json(filepath):
    """
    โหลดข้อมูลจากไฟล์ JSON ที่เป็น List of Dictionaries
    และแปลงเป็น Hugging Face Dataset
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            list_of_dicts = json.load(f)

        # ตรวจสอบว่าข้อมูลที่โหลดมาเป็น list และไม่ว่างเปล่า
        if not isinstance(list_of_dicts, list) or not list_of_dicts:
            raise ValueError(
                "ไฟล์ JSON ต้องเป็น list ที่มี dictionary อยู่ข้างใน")

        # ตรวจสอบ keys ใน dictionary แรก
        first_dict = list_of_dicts[0]
        if 'context' not in first_dict or 'question' not in first_dict or 'answer' not in first_dict:
            raise ValueError(
                "แต่ละ dictionary ใน list ต้องมี keys 'context', 'question', และ 'answer'")

        # แปลง list of dictionaries เป็น dictionary of lists
        # ซึ่งเป็นรูปแบบที่ Dataset.from_dict() ต้องการ
        processed_data = {
            'context': [item['context'] for item in list_of_dicts],
            'question': [item['question'] for item in list_of_dicts],
            'answer': [item['answer'] for item in list_of_dicts]
        }

        return Dataset.from_dict(processed_data)

    except FileNotFoundError:
        print(f"Error: ไม่พบไฟล์ที่ตำแหน่ง {filepath}")
        return None
    except json.JSONDecodeError:
        print(f"Error: ไฟล์ {filepath} ไม่ใช่ไฟล์ JSON ที่ถูกต้อง")
        return None
    except ValueError as e:
        print(f"Error: {e}")
        return None


def preprocess_text(text: str) -> str:
    text = text.lstrip('\ufeff')
    text = re.sub(r'\s+', ' ', text)
    allowed_chars_pattern = re.compile(r'[^ก-๛a-zA-Z0-9\s.,!?\'"-]')
    text = allowed_chars_pattern.sub('', text)
    text = text.lower()
    text = text.strip()

    return text