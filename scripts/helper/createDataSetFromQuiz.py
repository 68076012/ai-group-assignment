import os
from openai import OpenAI
from dotenv import load_dotenv
import re
from datasets import Dataset
import json
import random

load_dotenv()
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
model_name = "gpt-4.1-mini" 
filename = "data/trainData"
    
def createDataSet(stories):   
    dataSet = []
    createQuestionSystemPrompt = """คุณเป็นผู้ชายคิดคำถามจากนิทานเช่นที่ผู้ใช้ส่งมาให้แล้วส่งคืนเป็น context ของนิทาน, คำถาม, และคำตอบตามตามอย่างจำนวนทั้งหมด 10 คำถาม
    Ex.
    'context': 'หนูน้อยหมวกแดงเดินไปหาคุณยาย', 'question': 'หนูน้อยหมวกแดงไปหาใคร?', 'answer': 'คุณยาย'
    'context': 'หมาป่าซ่อนตัวอยู่ในป่า', 'question': 'ใครซ่อนตัวในป่า?', 'answer': 'หมาป่า'
    'context': 'หมูตัวสีชมพู', 'question': 'หมูตัวสีอะไร?', 'answer': 'ชมพู'"""
    for storie in stories:
        result = complete(storie, createQuestionSystemPrompt)
        pattern = r"'context':\s*'(.*?)',\s*'question':\s*'(.*?)',\s*'answer':\s*'(.*?)'"
        matches = re.findall(pattern, result, re.DOTALL)
        data = [{'context': c.strip(), 'question': q.strip(), 'answer': a.strip()} for c, q, a in matches]
        dataSet += data
    random.shuffle(dataSet)

    # Calculate the split index
    total_records = len(dataSet)
    train_split_index = int(total_records * 0.70)

    # Split the data
    train_data = dataSet[:train_split_index]
    test_data = dataSet[train_split_index:]

    # --- Saving the data ---

    # Save training data
    train_filename = 'trainData.json'
    with open(train_filename, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, ensure_ascii=False, indent=4)
    print(f"Training data saved to {train_filename} with {len(train_data)} records.")

    # Save test data
    test_filename = 'testData.json'
    with open(test_filename, 'w', encoding='utf-8') as f:
        json.dump(test_data, f, ensure_ascii=False, indent=4)
    print(f"Test data saved to {test_filename} with {len(test_data)} records.")
    

def complete(user_prompt, system_prompt):
    completion = openai_client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    )
    return completion.choices[0].message.content