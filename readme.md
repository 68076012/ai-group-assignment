# 🧠 AI สร้างชุดคำถามจากนิทาน

โปรเจกต์นี้เป็นระบบ AI ที่สามารถสร้างคำถามจากนิทานหรือเรื่องเล่าได้อัตโนมัติ โดยใช้โมเดล T5 ในการประมวลผลและสร้างภาษาไทย พร้อมการจัดการ tokenization ด้วย SentencePiece และอินเทอร์เฟซใช้งานผ่าน Gradio

---

## 🚀 คุณสมบัติหลัก

- รับอินพุตเป็นเนื้อเรื่องหรือนิทานภาษาไทย
- ประมวลผลและสร้างคำถาม (Question Generation) จากเนื้อเรื่อง
- ใช้โมเดล T5 ที่ผ่านการฝึกมาแล้ว (หรือ fine-tuned เพิ่มเติม)
- มีอินเทอร์เฟซใช้งานง่ายผ่าน Gradio

---

## 🛠️ เทคโนโลยีที่ใช้

- 🐍 **Python**
- 🔤 **SentencePiece** สำหรับ subword tokenization
- 🧠 **T5** โมเดล transformer สำหรับการสร้างคำถาม
- 🤗 **Transformers** และ **Datasets** จาก Hugging Face
- 🔥 **PyTorch** สำหรับการฝึกและใช้งานโมเดล
- 🌐 **Gradio** สำหรับอินเทอร์เฟซบนเว็บ
- 🔐 **OpenAI API** *(ถ้ามีใช้เสริม)* และการจัดการ token ด้วย `python-dotenv`

---

## 📦 การติดตั้ง

### 1. Clone Repository

```bash
git clone https://github.com/yourusername/ai-question-generator.git
cd ai-question-generator
pip install transformers torch datasets openai python-dotenv gradio sentencepiece
