# 📄 Policy Summary Assistant  
**AI-powered Insurance Policy Summarizer using NLP + TF-IDF + Gemini 2.5 Flash-Lite**

##  Overview  
Policy Summary Assistant is a high-performance insurance document intelligence system that converts **unstructured PDF/TXT policy documents into accurate, policy-specific summaries within seconds**.  

<img width="1508" height="673" alt="image" src="https://github.com/user-attachments/assets/ed130642-4cb0-45a3-bd58-d1c11f92fa39" />
<img width="1390" height="668" alt="image" src="https://github.com/user-attachments/assets/93ac0d45-fdab-4f90-aefd-a3cfb4f8caf6" />

The system extracts text, identifies key insurance terms, and generates a structured 5-section policy summary using **Google Gemini 2.5 Flash-Lite**, optimized with caching and batching for **3–6× faster performance**.

---

## 📌 Workflow Diagram  
<img width="370" height="434" alt="Screenshot 2025-12-03 160357" src="https://github.com/user-attachments/assets/5bab431d-bdeb-4ec7-988f-8df186e757b6" />


---

##  Project Details  
| Field | Description |
|------|-------------|
| **Use Case Title** | Policy Summary Assistant |
| **Model Type** | AI-powered Insurance Policy Summarizer |
| **Category** | Gen AI + NLP + InsurTech |


---

##  Key Features
- Upload **PDF/TXT** insurance documents  
- Extract text using **PyMuPDF**  
- Clean + tokenize sentences with **spaCy NLP**  
- Identify top keywords with **TF-IDF**  
- Generate fast & structured policy summaries using **Gemini 2.5 Flash-Lite**  
- Display “Summit View” → **Top 12 policy keywords**  
- Download **policy-specific text summaries**  
- 3–6× speedup vs LangChain due to optimized batching + caching  

---

##  System Architecture (From PDF Summary)  

### **1. Document Extraction — PyMuPDF**
- Precisely extracts raw text  
- Maintains layout and segmentation  
- Handles long insurance policy structures  

### **2. NLP Preprocessing — spaCy**
- Sentence tokenization  
- Linguistic normalization  
- Prepares clean text batches  

### **3. Keyword Extraction — TF-IDF (scikit-learn)**
- Identifies top 12 most relevant terms  
- Ensures the generated summary stays grounded in policy content  

### **4. Summary Generation — Gemini 2.5 Flash-Lite**
- Processes context-optimized batches  
- Generates 5-section Markdown insurance summary  
- Uses caching for high throughput  

### **5. Output**
- **Summit View** (Top 12 TF-IDF terms)  
- **Downloadable .txt file** with policy-specific summary  

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|------------|
| **Frontend** | Streamlit |
| **NLP** | spaCy (Sentence tokenization) |
| **Keyword Extraction** | scikit-learn TF-IDF |
| **Document Parsing** | PyMuPDF |
| **LLM Engine** | Gemini 2.5 Flash-Lite API |
| **Output** | Markdown + Downloadable .txt |

---

## 🎥 Demo Video

You can view/download the demo here:

[Click here to watch the demo] https://drive.google.com/file/d/1VcEvVtRe-uQMPCnHWmOjD1i7Scyg8ewi/view?usp=drive_link

---
## 📦 Installation

```bash
git clone <your-repo-url>
cd policy-summary-assistant
```
# Create venv
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```
# How to Run
```bash
python app.py
```
