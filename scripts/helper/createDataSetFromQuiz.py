import os
from openai import OpenAI
from dotenv import load_dotenv
import re
from datasets import Dataset
import json
from transformers import T5Tokenizer

load_dotenv()
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
model_name = "gpt-4.1-mini"
filename = "data/trainData"
tokenizer = T5Tokenizer.from_pretrained("t5-base")


def createDataSet(stories):
    result = []
    createQuestionSystemPrompt = """คุณเป็นผู้ชายคิดคำถามจากนิทานเช่นที่ผู้ใช้ส่งมาให้แล้วส่งคืนเป็น input: นิทาน, เป้าหมาย: คำถามและคำตอบตามตามอย่างจำนวนทั้งหมด 3 คำถาม และให้ส่งคืนเฉพาะ JSON เท่านั้น โดยอยู่ใน code block เช่น
    ```json
            [{
                "Input" : *** user input ***
                "target": 
                    {
                    "question": "หนูบ้านรู้สึกอย่างไรเกี่ยวกับการใช้ชีวิตในชนบท?",
                    "answer": "หนูบ้านคิดว่าชีวิตในชนบทน่าเบื่อ และอาหารไม่อร่อยเท่าในเมือง"
                    },
                    {
                    "question": "เหตุการณ์ใดทำให้หนูนาอยากกลับบ้านที่ชนบท?",
                    "answer": "การที่หนูนาเกือบถูกแมวไล่จับ ทำให้มันรู้สึกว่าการใช้ชีวิตในเมืองอันตราย"
                    },
                    {
                    "question": "นิทานเรื่องนี้สอนให้เรารู้เรื่องอะไร?",
                    "answer": "นิทานสอนให้รู้ว่า ไม่มีที่ใดอบอุ่นใจเท่าบ้านของเรา"
                    }
                
            }, "Input" : *** user input ***, "target": [...]] ```
            
    ในส่วนของ user input ที่ฉัน highlight *** ไว้ให้ใส่ user input ที่เป็นนิทานที่ user ส่งมา"""

    for storie in stories:
        chunks = split_story_by_tokens(preprocess_text(storie))

        for chunk in chunks:
            raw_response = complete(chunk, createQuestionSystemPrompt)
            print("Raw GPT Response:", raw_response)   # debug ดูข้อความที่ได้
            json_text = extract_json_from_response(raw_response)
            print("Extracted JSON Text:", json_text)  # debug ดูข้อความที่ได้จาก regex
            response = safe_json_parse(json_text)
            result.extend(response)
    with open("data/data_train.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)


def split_story_by_tokens(text, max_tokens=450):
    if tokenizer is None:
        raise ValueError("ต้องระบุ tokenizer ที่ใช้ในการนับ token")

    sentences = re.split(r'(?<=[.?!])\s+', text)
    chunks = []
    current_chunk = []
    current_tokens = 0

    for sentence in sentences:
        token_count = len(tokenizer.encode(sentence))

        if token_count > max_tokens:
            if current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_tokens = 0

            words = sentence.split()
            temp_chunk = []
            temp_tokens = 0
            for word in words:
                word_tokens = len(tokenizer.encode(word))
                if temp_tokens + word_tokens > max_tokens:
                    chunks.append(" ".join(temp_chunk))
                    temp_chunk = [word]
                    temp_tokens = word_tokens
                else:
                    temp_chunk.append(word)
                    temp_tokens += word_tokens
            if temp_chunk:
                chunks.append(" ".join(temp_chunk))
        else:
            if current_tokens + token_count > max_tokens:
                chunks.append(" ".join(current_chunk))
                current_chunk = [sentence]
                current_tokens = token_count
            else:
                current_chunk.append(sentence)
                current_tokens += token_count

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def complete(user_prompt, system_prompt):
    completion = openai_client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    )
    return extract_json_from_response(completion.choices[0].message.content)

def safe_json_parse(text):
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        print("JSON decode error:", e)
        print("Raw text response:\n", text)
        raise e

def extract_json_from_response(text):
    # ใช้ .*? แบบ non-greedy บางครั้งอาจตัดไม่หมด ลองเปลี่ยนเป็น greedy .* 
    match = re.search(r"```json\s*(.*)\s*```", text, re.DOTALL)
    return match.group(1) if match else text

def preprocess_text(text: str) -> str:
    text = text.lstrip('\ufeff')
    text = re.sub(r'\s+', ' ', text)
    allowed_chars_pattern = re.compile(r'[^ก-๛a-zA-Z0-9\s.,!?\'"-]')
    text = allowed_chars_pattern.sub('', text)
    text = text.lower()
    text = text.strip()

    return text
