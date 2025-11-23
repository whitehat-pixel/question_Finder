# 📚 question_finder  

A local ChatGPT-style chatbot that finds **Previous Year Questions (PYQ)** from PDF files using **TF-IDF & Cosine Similarity**.  
It works completely **offline** and supports **multiple subjects** like OS, DCN, SE, etc.  

---

## 🚀 Features  
✔ ChatGPT-style UI  
✔ Subject dropdown (OS / DCN / SE / etc.)  
✔ Searches real PDF files  
✔ Runs locally – no internet needed  
✔ Flask backend + HTML/CSS frontend  

---

## 🛠 Tech Used  
| Layer | Technology |
|------|------------|
| Backend | Python + Flask |
| UI | HTML, CSS, JavaScript |
| ML | TF-IDF + Cosine Similarity |
| PDF Reading | pdfplumber |

---

## 📂 Folder Structure  

question_finder/
│── app.py
│── requirements.txt
│── README.md
│── .gitignore
│── pdfs/
│ ├── os/
│ └── dcn/
│── templates/
│ └── chat.html
│── static/
│ └── style.css
---

## 🧪 Setup & Run  

```bash
pip install -r requirements.txt
python app.py
4️⃣ Open in browser

👉 http://127.0.0.1:5000
🧪 Example Topics to Search

Try entering:

system calls  
demand paging  
cpu scheduling  
virtual memory  
round robin scheduling  
directory structure  
paging vs segmentation
📌 How to Add New Subject

Create a new subject folder inside pdfs/:

pdfs/
 ├── os/
 ├── dcn/
 ├── se/
 └── dbms/


Then add your PDFs inside it:

pdfs/dbms/dbms_2022.pdf
pdfs/dbms/dbms_2023.pdf🚀 Future Improvements

AI answer generation (OpenAI / Gemini API)

Export results to PDF

Student login system

Upload PDF from UI

Deploy online (Render / Vercel / Railway)

Mobile responsive UI
❤️ Author

Developed by: Purushottam Bairagi
Built to help students prepare better for exams!
Let’s make studying smarter 📚✨
