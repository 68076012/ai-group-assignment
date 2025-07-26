I Story Quiz Generator
This project develops an AI model capable of automatically generating multiple-choice quizzes from fables or short stories. It leverages Large Language Models (LLMs) for question and answer generation and fine-tunes a T5 model for the task.

🌟 Features
Automated Data Preparation: Loads story data from text files and, optionally, uses an LLM (like OpenAI's GPT) to generate question-answer pairs for training.

Model Training: Fine-tunes a T5 (Text-to-Text Transfer Transformer) model on your custom question-answer dataset.

Quiz Generation: Generates new questions and multiple-choice options (including distractors) from unseen story texts.

Modular Design: Code is organized into separate scripts for data preparation, model training, and quiz generation, making it easy to understand and extend.

🚀 Getting Started
Follow these steps to set up and run the project on your local machine.

Prerequisites
Before you begin, ensure you have the following installed:

Python 3.10+ (Python 3.11 or 3.12 is recommended for better compatibility with libraries like sentencepiece).

pip (Python package installer).

Git (for cloning the repository).

CMake: Required for building the sentencepiece library.

Download from cmake.org/download.

Crucially, during installation, ensure you select "Add CMake to the system PATH for all users" or "Add CMake to the system PATH for the current user".

After installation, restart your terminal/command prompt and verify by typing cmake --version.

Installation
Clone the repository:

git clone <repository_url>
cd <repository_name> # e.g., cd group-assignment

Create and activate a Virtual Environment (Recommended):
This isolates your project dependencies from your system's Python installation.

python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

Install Python dependencies:

pip install -r requirements.txt

(If requirements.txt is not provided, you can generate one or install manually):

pip install transformers torch datasets openai python-dotenv sentencepiece

Troubleshooting sentencepiece installation: If you encounter subprocess-exited-with-error or FileNotFoundError: [WinError 2] related to cmake, ensure CMake is correctly installed and its path is in your system's PATH environment variable as described in the Prerequisites. Also, try clearing pip cache: python -m pip cache purge.

Set up OpenAI API Key:

Create a file named .env in the root directory of your project (e.g., group-assignment/.env).

Add your OpenAI API key to this file:

OPENAI_API_KEY="sk-YOUR_OPENAI_API_KEY_HERE"

Replace "sk-YOUR_OPENAI_API_KEY_HERE" with your actual key.

📁 Project Structure
project_quiz_ai/
├── data/
│   ├── tales/        # Your 40 story .txt files (e.g., tale1.txt, tale2.txt)
│   │   ├── tale1.txt
│   │   └── ...
│   ├── trainData.json # JSON file for training data (context, question, answer)
│   └── testData.json  # JSON file for testing data (context, question, answer)
├── scripts/
│   ├── helper/
│   │   └── helper.py  # Utility functions, e.g., load_data_from_json
│   ├── data_preparation.py # Handles loading and splitting data
│   ├── train_model.py      # Contains model training logic
│   └── generate_quiz.py    # Logic for generating quizzes
├── models/             # Directory where trained models will be saved
│   └── (trained_model_files)
├── .env                # Stores environment variables like API keys
├── main.py             # Main entry point for the application
└── README.md           # This file

💡 Usage
The main.py script orchestrates the entire process.

Prepare your Data:

Place your 40 story .txt files inside the data/tales/ directory.

Ensure your trainData.json and testData.json files are in the data/ directory. These JSON files should contain question-answer pairs derived from your stories.

Expected trainData.json / testData.json format:
The JSON files should be a list of dictionaries, where each dictionary represents a question-answer pair:

[
  {
    "context": "หนูน้อยหมวกแดงเดินไปหาคุณยาย",
    "question": "หนูน้อยหมวกแดงไปหาใคร?",
    "answer": "คุณยาย"
  },
  {
    "context": "หมาป่าซ่อนตัวอยู่ในป่า",
    "question": "ใครซ่อนตัวในป่า?",
    "answer": "หมาป่า"
  },
  {
    "context": "พ่อแม่ของนกสาลิกาไม่ได้ดุแต่ให้กินหนอนและสอนวิธีซ่อนลูกโอ๊ก",
    "question": "พ่อแม่ของนกสาลิกามีปฏิกิริยาอย่างไรเมื่อนกสาลิกากลับมา?",
    "answer": "ไม่ได้ดุแต่ให้กินหนอนและสอนวิธีซ่อนลูกโอ๊ก"
  }
]

Note: The data_preparation.py script currently loads data directly from trainData.json and testData.json. If you want to generate Q&A pairs from the .txt files using OpenAI, you'll need to uncomment and adjust the relevant parts in data_preparation.py and main.py.

Run the Main Script:
Navigate to the root directory of your project (group-assignment/) in your terminal and run:

python main.py

The main.py script will:

Load your trainData.json and testData.json.

(Commented by default) Initiate the model training process using train_model.py. You will need to uncomment the train_model.train_t5_model(...) line in main.py when you are ready to train the model.

(Commented by default) After training, it can generate quizzes using generate_quiz.py. Uncomment the relevant lines in main.py to enable quiz generation.

🤝 Contributing
Contributions are welcome! If you have suggestions for improvements or find bugs, please open an issue or submit a pull request.

📄 License
This project is licensed under the MIT License - see the LICENSE file for details.