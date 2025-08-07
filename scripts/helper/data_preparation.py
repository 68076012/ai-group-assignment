# scripts/data_preparation.py

import os
from typing import List
import instructor
from openai import BaseModel, OpenAI
import json
from dotenv import load_dotenv

# Define the inner structure for question/answer
class Target(BaseModel):
    question: str
    answer: str

# Define each QA context item
class QAItem(BaseModel):
    Context: str
    target: Target

load_dotenv()
client = instructor.from_openai(OpenAI(api_key=os.getenv("OPENAI_API_KEY")))
model_name = "gpt-4-turbo"  # แนะนำให้ใช้ gpt-4-turbo เพื่อความน่าเชื่อถือ
# เปลี่ยน system_prompt ให้สอดคล้องกับ Pydantic model
createQuestionSystemPrompt = """คุณเป็นผู้ช่วยที่เชี่ยวชาญในการสร้างชุดข้อมูลสำหรับโมเดล QA (Question-Answering)
ภารกิจของคุณคือสร้างคำถามและคำตอบ 4 ชุดจากนิทานที่ผู้ใช้ให้มา
ข้อกำหนด:
- Context: สรุปเนื้อหาจากนิทานที่ใช้ในการสร้างคำถาม-คำตอบ
- question: คำถามจากเนื้อหาใน Context
- answer: คำตอบจากเนื้อหาใน Context
- ทุก Context, question, และ answer ต้องอ้างอิงจากเนื้อหาต้นฉบับเท่านั้น
- ความยาวของ Context, question, และ answer รวมกันต้องไม่เกิน 300 คำ
- สร้างคำถามและคำตอบให้ครบ 4 ชุดเสมอ"""


def load_stories_from_folder(folder_path):
    """
    Loads story content from all .txt files in the folder.
    """
    stories = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            filepath = os.path.join(folder_path, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                stories.append(f.read())
    return stories


def complete(user_prompt, system_prompt):
    # instructor จะจัดการเรื่อง JSON output ให้เอง
    return client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_model=List[QAItem],
        temperature=0.0,
        timeout=60.0
    )


def createDataSet(stories: List[str]):
    """
    Creates a dataset of questions and answers from a list of stories,
    saving the results to a JSON file.
    """
    output_path = "../../data/data_train.json"
    all_qas = []

    for index, story in enumerate(stories):
        print(f"Processing story {index + 1} of {len(stories)}...")
        try:
            qa_items = complete(story, createQuestionSystemPrompt)
            # เพิ่มแต่ละ item เข้าไปใน list รวม
            for item in qa_items:
                all_qas.append(item.dict())
        except Exception as e:
            print(f"Warning: Failed to process story {index + 1}. Error: {e}")
            continue

    # เขียนข้อมูลทั้งหมดลงในไฟล์ JSON เพียงครั้งเดียวตอนจบ
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_qas, f, ensure_ascii=False, indent=2)

    print(f"Dataset creation complete! Data saved to {output_path}")


if __name__ == '__main__':
    stories_folder = "../../data/tales"
    stories = load_stories_from_folder(stories_folder)
    if stories:
        createDataSet(stories)
    else:
        print(f"No .txt files found in '{stories_folder}'. Please check the folder path and contents.")