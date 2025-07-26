# scripts/train_model.py

from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments, DataCollatorForSeq2Seq
from datasets import Dataset

def preprocess_function(examples, tokenizer):
    """
    แปลงข้อมูลข้อความเป็นตัวเลข (tokens) ที่โมเดลเข้าใจ
    """
    # ตัวอย่างนี้เป็นการฝึกสอนเพื่อสร้างคำถามจาก context
    # ถ้าคุณต้องการสร้างคำตอบจากคำถาม คุณต้องเปลี่ยน 'input' และ 'target'
    inputs = [f"generate question: {context}" for context in examples['context']]
    model_inputs = tokenizer(inputs, max_length=512, truncation=True)

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples['question'], max_length=128, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def train_t5_model(train_dataset, val_dataset, model_output_dir):
    model_name = "t5-small"
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    
    tokenized_train_dataset = train_dataset.map(lambda examples: preprocess_function(examples, tokenizer), batched=True)
    tokenized_val_dataset = val_dataset.map(lambda examples: preprocess_function(examples, tokenizer), batched=True)

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    training_args = TrainingArguments(
        output_dir=model_output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=500,
        weight_decay=0.01,
        eval_strategy="epoch",
        # เพิ่มบรรทัดนี้: ตั้งค่า save_strategy ให้เป็น "epoch"
        save_strategy="epoch", # <--- แก้ไขตรงนี้
        logging_dir='./logs',
        report_to="none", 
        load_best_model_at_end=True, 
        metric_for_best_model="eval_loss", 
        greater_is_better=False, 
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_val_dataset,
        tokenizer=tokenizer, 
        data_collator=data_collator, 
    )
    
    trainer.train()
    trainer.save_model(model_output_dir)
    tokenizer.save_pretrained(model_output_dir)

# if __name__ == '__main__':
#     # สมมติว่าคุณได้บันทึก datasets ไว้แล้วใน data_preparation.py
#     # หรือคุณสามารถสร้างชุดข้อมูลขึ้นมาใหม่ที่นี่
#     dummy_data = {'context': ['...', '...'], 'question': ['...','...'], 'answer': ['...','...']}
#     dummy_train_ds = Dataset.from_dict(dummy_data)
#     dummy_val_ds = Dataset.from_dict(dummy_data)
    
#     model_output_dir = '../models'
#     train_t5_model(dummy_train_ds, dummy_val_ds, model_output_dir)
#     print(f"ฝึกสอนโมเดลเสร็จสิ้นและบันทึกที่ {model_output_dir}")