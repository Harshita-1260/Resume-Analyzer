import streamlit as st 
import matplotlib.pyplot as plt 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2
import re 
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk .tokenize import word_tokenize
from nltk import pos_tag

#dowloads some nltk resources
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger_eng')

#page configuration
st.set_page_config(page_title="RESUME JOB MATCH SCORER", page_icon="📜", layout="wide")
st.title("RESUME JOB MATCH SCORER")
st.markdown("""Upload your resume (PDF format) and paste the job description to see how well they match!
        This tool uses ** TDF-IDF ** and ** Cosine Similarity ** to analyze your resume against job requirements""")

with st.sidebar:
    st.header("About")
    st.info("""
     This applictaion helps :
        - Measures how your resume matches a job description 
        - Identify important job keywords
        - Improve your resume based on missing terms
    """
    )
    st.header("How it works")
    st.write("""
        1. Upload your resume in PDF format
        2. Paste the job description
        3. Click  **Analyze Match**
        4. Review score & suggestions     
    """)

#create function for text from PDF 
def extract_text_from_pdf(pdf_file):
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        st.error(f"Error extracting text from PDF: {e}")
        return ""
    
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text   

def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text)
    filtered_words = [word for word in words if word not in stop_words]
    return ' '.join(filtered_words)

def calculate_similarity(resume_text, job_description):
   resume_processed=remove_stopwords(clean_text(resume_text))
   job_processed=remove_stopwords(clean_text(job_description))
   vectorizer = TfidfVectorizer()
   tfidf_matrix = vectorizer.fit_transform([resume_processed, job_processed])
   score= cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]*100
   return round(score, 2),resume_processed, job_processed  




#main dashboard
def main():
  pdf_file = st.file_uploader("Upload your resume (PDF format)", type=["pdf"])
  job_description = st.text_area("Paste the job description here", height=200)

  if st.button("Analyze Resume", key ="analyze_resume") and pdf_file and job_description:
        if not pdf_file:
            st.warnings("Please upload a valid PDF file.")
            return
        if not job_description.strip():
            st.warnings("Please paste the job description.")
            return
        with st.spinner("Analyzing..."):
            resume_text = extract_text_from_pdf(pdf_file)
            if not resume_text:
                st.error("Failed to extract text from the uploaded PDF. Please check the file and try again.")
                return
            #calculate similarity score
            similarity_score, resume_processed, job_processed = calculate_similarity(resume_text, job_description)

            #results
            st.subheader("Results")
            st.metric("Match Score", f"{similarity_score:.2f}%")

#gauge chart for match score
            fig, ax = plt.subplots(figsize=(6, 0.5))
            colors = ['red', 'orange', 'yellow', 'green']
            color_index = min(int(similarity_score//33), 2)
            ax.barh([0], [similarity_score], color=colors[color_index])
            ax.set_xlim(0, 100)
            ax.set_xlabel("Match Score")
            ax.set_yticks([])
            ax.set_title("Resume Analysis")
            st.title("Match Score Gauge")
            st.pyplot(fig)

            if similarity_score < 40:
                st.warning("Your resume has a low match score. Consider revising it to better align with the job description.")
            elif similarity_score < 70:
                st.info("Your resume has a moderate match score. You may want to improve it for better alignment with the job description.")
            else:
                st.success("Your resume has a high match score. It aligns well with the job description.")

if __name__ == "__main__":
    main()
        
