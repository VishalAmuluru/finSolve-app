# finSolve-app

A smart and unbiased loan advisor chatbot powered by LangChain, FAISS, OpenAI, and deployed using Streamlit.

---

## 1. Problem Statement

When people search for loan options online, they often get **biased or incomplete results**.  
Most platforms promote specific banks or limited offers, making it hard for users to find the **best loan deal** for their needs.

---

## 2. Project Goal

To solve this, I’ve built a **smart, unbiased loan advisor chatbot**.  
It allows users to ask natural language questions like:

- “Which bank has the lowest home loan interest rate?”
- “What personal loans are available for self-employed people in Hyderabad?”

Instead of simple keyword matching, the chatbot **understands the meaning** behind the question and gives the **most relevant answer** using real data.

---

## 3. Tech Stack Used

- **LangChain** – for building the QA pipeline  
- **FAISS** – for fast semantic vector search  
- **OpenAI API** – for embeddings and GPT-based language understanding  
- **Pandas** – for loading and formatting the dataset  
- **Streamlit** – for deploying the application with a simple UI  

---

## 4. How It Works

1. Loads a dataset of over **10,000 loan offers**.
2. Formats and splits the data into smaller text chunks.
3. Converts the chunks into **embeddings** using OpenAI’s `text-embedding-3-small` model.
4. Stores the embeddings in a **FAISS index** for fast vector search.
5. When a user asks a question:
   - FAISS retrieves the most relevant loan-related chunks.
   - GPT-3.5 Turbo processes those chunks and generates a human-like response.

---

## 5. Deployment

The chatbot is deployed using **Streamlit** to provide a **simple, interactive web interface**.  
Users can ask loan-related queries directly from the browser without needing any technical knowledge.  
This makes it ideal for demos, testing, and real-world usage.

---

## 6. Sample Output

**User Query:**  
> “Which bank gives the lowest home loan interest?”

**Chatbot Response:**  
> “SBI offers the lowest rate at 7.3%, followed by HDFC at 7.5%.”

This shows the system can understand the user's intent and return **reliable, unbiased answers**.

---

## 7. Future Scope

Planned enhancements include:

- Adding more filters (city, loan type, amount range)
- Using real-time data via **bank APIs**
- Expanding to support **multiple cities and languages**
- Improving the UI for mobile and tablet users

---

## 8. Author

**Vishal Krishna**  
Student, CSE  
Amrita Vishwa Vidyapeetham

---

## 9. License

This project is for academic and learning purposes. For commercial use, appropriate licenses and APIs must be obtained.


