
# CSU Global Academic Chatbot 🎓

This is a closed-domain NLP chatbot designed to answer questions related to Colorado State University Global (CSU Global).

## Features
- Uses NLP (TF-IDF + cosine similarity)
- Answers academic questions
- Covers:
  - Academic Calendar
  - Admissions
  - Programs
  - Tuition
  - Student Services
- Uses real CSU Global website data

## Technologies Used
- Python
- Streamlit
- Scikit-learn
- BeautifulSoup
- Requests

## How to Run

1. Clone the repository
git clone https://github.com/YOUR_USERNAME/CSC525
cd csu-global-chatbot

2. Install dependencies
pip install -r requirements.txt

3. Run the chatbot
streamlit run app.pyShow more lines

4. Open in browser
http://localhost:8501

# Notes

This is a closed-domain chatbot.
It only answers questions about CSU Global.
For unsupported questions, it returns a fallback response.
