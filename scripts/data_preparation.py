# scripts/data_preparation.py

import os
from datasets import Dataset
import json


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


def get_qa_dataset():
    """
    ใช้โมเดลภาษาขนาดใหญ่ (LLM) หรือวิธีการอื่นเพื่อสร้างคู่คำถาม-คำตอบ
    ตัวอย่างนี้เป็นการจำลองการสร้างข้อมูล
    """
    qa_pairs = load_data_from_json("data/trainData.json")
    print(qa_pairs)

    return Dataset.from_dict({
        'context': [item['context'] for item in qa_pairs],
        'question': [item['question'] for item in qa_pairs],
        'answer': [item['answer'] for item in qa_pairs]
    })


def split_dataset(dataset, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """
    แบ่งชุดข้อมูลออกเป็นชุดฝึกสอน (train), ตรวจสอบ (validation) และทดสอบ (test)
    """
    train_size = int(len(dataset) * train_ratio)
    val_size = int(len(dataset) * val_ratio)

    train_dataset = dataset.select(range(train_size))
    val_dataset = dataset.select(range(train_size, train_size + val_size))
    test_dataset = dataset.select(range(train_size + val_size, len(dataset)))

    return train_dataset, val_dataset, test_dataset


if __name__ == '__main__':
    stories_folder = '../data'
    stories = load_stories_from_folder(stories_folder)

    if stories:
        dataset = get_qa_dataset(stories)
        train_ds, val_ds, test_ds = split_dataset(dataset)

        print(f"ขนาดชุดข้อมูลฝึกสอน: {len(train_ds)}")
        print(f"ขนาดชุดข้อมูลตรวจสอบ: {len(val_ds)}")
        print(f"ขนาดชุดข้อมูลทดสอบ: {len(test_ds)}")

        # คุณสามารถบันทึก datasets เหล่านี้เพื่อใช้ใน train_model.py ได้
        # train_ds.save_to_disk('train_dataset')
        # val_ds.save_to_disk('val_dataset')
        # test_ds.save_to_disk('test_dataset')


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
            raise ValueError("ไฟล์ JSON ต้องเป็น list ที่มี dictionary อยู่ข้างใน")

        # ตรวจสอบ keys ใน dictionary แรก
        first_dict = list_of_dicts[0]
        if 'context' not in first_dict or 'question' not in first_dict or 'answer' not in first_dict:
            raise ValueError("แต่ละ dictionary ใน list ต้องมี keys 'context', 'question', และ 'answer'")

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