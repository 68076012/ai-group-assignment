# main.py

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts'))

import data_preparation
import train_model
import generate_quiz
from scripts.helper import createDataSetFromQuiz

if __name__ == '__main__':
    # print("ขั้นตอนที่ 1: เตรียมข้อมูล")
    
    stories_folder = 'data/tales' 
    
    if not os.path.exists(stories_folder):
        print(f"Error: ไม่พบโฟลเดอร์ '{stories_folder}' กรุณาตรวจสอบเส้นทาง")
        sys.exit()

    stories = data_preparation.load_stories_from_folder(stories_folder)
    createDataSetFromQuiz.createDataSet(stories[0:1])
    
    if stories:
        # dataset = data_preparation.get_qa_dataset()
        # train_ds, val_ds, test_ds = data_preparation.split_dataset(dataset)
        print(f"เตรียมข้อมูลเสร็จสิ้น")
    
    # print("\nขั้นตอนที่ 2: ฝึกสอนโมเดล")
    # model_output_dir = 'models'
    
    # # # # ลบคอมเมนต์และเปิดใช้งานโค้ดนี้เมื่อคุณพร้อมจะฝึกสอน
    # # train_model.train_t5_model(train_ds, val_ds, model_output_dir)
    # # # print("การฝึกสอนโมเดลถูกข้ามไป (โค้ดอยู่ในคอมเมนต์)")
    
    # print("\nขั้นตอนที่ 3: สร้าง Quiz อัตโนมัติ")
    # if os.path.exists(model_output_dir):
    #     # ลบคอมเมนต์และเปิดใช้งานโค้ดนี้เมื่อคุณพร้อมจะสร้าง Quiz
    #     story_to_quiz = stories[5] # ตัวอย่าง: ใช้เรื่องแรกจากชุดข้อมูล
    #     tokenizer, model = generate_quiz.load_trained_model(model_output_dir)
    #     question, answer = generate_quiz.generate_question_and_answer(tokenizer, model, story_to_quiz)
    #     quiz = generate_quiz.create_multiple_choice_quiz(question, answer, story_to_quiz)
    #     print(quiz)
    #     print("สามารถสร้าง Quiz ได้เมื่อโมเดลถูกฝึกสอนแล้ว")
    # else:
    #     print("ไม่พบโมเดลที่ฝึกสอนแล้ว กรุณาฝึกสอนโมเดลก่อน")