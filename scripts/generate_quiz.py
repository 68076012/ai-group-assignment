# scripts/generate_quiz.py

from transformers import T5Tokenizer, T5ForConditionalGeneration

def load_trained_model(model_path):
    """
    โหลดโมเดลและ tokenizer ที่ฝึกสอนแล้ว
    """
    tokenizer = T5Tokenizer.from_pretrained(model_path)
    model = T5ForConditionalGeneration.from_pretrained(model_path)
    return tokenizer, model

def generate_question_and_answer(tokenizer, model, story_text):
    """
    สร้างคำถามและคำตอบจากเนื้อหาของนิทาน
    """
    input_text = f"generate question: {story_text}"
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids
    
    outputs = model.generate(input_ids)
    question = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # ในส่วนนี้คุณอาจต้องมีโมเดลอีกตัวเพื่อสร้างคำตอบ หรือใช้ heuristic (กฎ)
    # เช่น คำตอบอาจเป็นส่วนหนึ่งของประโยคใน story_text
    
    return question, "ตัวอย่างคำตอบ" # ต้องปรับปรุงส่วนนี้

def create_multiple_choice_quiz(question, correct_answer, story_text):
    """
    สร้าง Quiz แบบปรนัยโดยเพิ่มคำตอบหลอก
    """
    # วิธีง่ายๆ: หาประโยคอื่นๆ จากเรื่องเล่าเป็นตัวเลือกหลอก
    sentences = story_text.split('.')
    distractors = [s.strip() for s in sentences if s.strip() != correct_answer][:3]
    
    options = distractors + [correct_answer]
    # สับเปลี่ยนลำดับของตัวเลือก
    import random
    random.shuffle(options)
    
    quiz_item = {
        "question": question,
        "options": options,
        "correct_answer": correct_answer
    }
    return quiz_item

if __name__ == '__main__':
    model_path = '../models'
    try:
        tokenizer, model = load_trained_model(model_path)
        
        new_story = "หนูน้อยหมวกแดงเดินทางผ่านป่าที่มืดมิดและน่ากลัว เธอพบกับหมาป่าเจ้าเล่ห์ที่แกล้งทำเป็นมิตร"
        question, answer = generate_question_and_answer(tokenizer, model, new_story)
        
        quiz = create_multiple_choice_quiz(question, "หมาป่าเจ้าเล่ห์", new_story)
        print("Quiz ที่สร้างขึ้น:")
        print(f"คำถาม: {quiz['question']}")
        for i, option in enumerate(quiz['options']):
            print(f"  {chr(65+i)}. {option}")
        print(f"คำตอบที่ถูกต้อง: {quiz['correct_answer']}")

    except Exception as e:
        print(f"เกิดข้อผิดพลาด: {e}")
        print("ตรวจสอบว่าคุณได้ฝึกสอนโมเดลและบันทึกไว้ในโฟลเดอร์ '../models' แล้ว")