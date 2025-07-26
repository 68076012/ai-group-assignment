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
    createQuestionSystemPrompt = """คุณเป็นผู้ชายคิดคำถามจากนิทานเช่นที่ผู้ใช้ส่งมาให้แล้วส่งคืนเป็น input dialog ของนิทาน, เป้าหมาย, คำถามและคำตอบตามตามอย่างจำนวนทั้งหมด 10 คำถาม
    Ex.
            {
                "input": "tale: '''user input'''",
                "target": "question: หนูนายินดีจะอยู่อย่างสงบสุขไหม? answer: ใช่"
            },
            {
                "input": "tale: '''user input'''",
                "target": "question: หนูบ้านอุทานว่าอะไร? answer: กินอาหารกลางวันที่ต่างจังหวัด--น่าสนใจเป็นที่สุด!"
            }
            
    ในส่วนของ user input ที่ฉัน highlight *** ไว้ให้ใส่ user input ที่เป็นนิทานที่ user ส่งมา"""
    for storie in stories:
        result = complete(storie, createQuestionSystemPrompt)
        # pattern = r"'context':\s*'(.*?)',\s*'question':\s*'(.*?)',\s*'answer':\s*'(.*?)'"
        # matches = re.findall(pattern, result, re.DOTALL)
        # data = [{'context': c.strip(), 'question': q.strip(), 'answer': a.strip()} for c, q, a in matches]
        dataSet += parse_custom_json_string(result)
        
    train_filename = 'trainData.json'
    with open(train_filename, 'w', encoding='utf-8') as f:
        json.dump(dataSet, f, ensure_ascii=False, indent=4)
    # random.shuffle(dataSet)

    # # Calculate the split index
    # total_records = len(dataSet)
    # train_split_index = int(total_records * 0.70)

    # # Split the data
    # train_data = dataSet[:train_split_index]
    # test_data = dataSet[train_split_index:]

    # # --- Saving the data ---

    # # Save training data
    # train_filename = 'trainData.json'
    # with open(train_filename, 'w', encoding='utf-8') as f:
    #     json.dump(train_data, f, ensure_ascii=False, indent=4)
    # print(f"Training data saved to {train_filename} with {len(train_data)} records.")

    # # Save test data
    # test_filename = 'testData.json'
    # with open(test_filename, 'w', encoding='utf-8') as f:
    #     json.dump(test_data, f, ensure_ascii=False, indent=4)
    # print(f"Test data saved to {test_filename} with {len(test_data)} records.")
    
def parse_custom_json_string(raw_string):
    # ขั้นตอนที่ 1: ลบ prefix '{\n  "input": ' และ suffix '\n}'
    if raw_string.startswith('{') and '"input":' in raw_string:
        start_index = raw_string.find('"input":') + len('"input": ')
        end_index = raw_string.rfind('}')
        json_like_string = raw_string[start_index:end_index].strip()

        # ขั้นตอนที่ 2: ลบเครื่องหมาย ' หรือ ''' รอบข้อความ
        if json_like_string.startswith(("'''", '"', "'")):
            json_like_string = json_like_string.strip('\'"')

        # ขั้นตอนที่ 3: decode escape characters เช่น \n, \t ฯลฯ
        decoded_string = json_like_string.encode('utf-8').decode('unicode_escape')

        # ขั้นตอนที่ 4: สร้าง JSON จากข้อความที่แยกเป็นบรรทัด
        result = {}
        lines = decoded_string.strip().splitlines()
        for line in lines:
            if "question:" in line and "answer:" in line:
                q_part, a_part = line.split("answer:", 1)
                question = q_part.replace("question:", "").strip()
                answer = a_part.strip()
                result[question] = answer

        return result
    else:
        raise ValueError("รูปแบบ input ไม่ถูกต้อง")

def complete(user_prompt, system_prompt):
    completion = openai_client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    )
    return completion.choices[0].message.content