import os
import PyPDF2
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

folder_path= r"C:\Users\Prasad Thosar\OneDrive\Desktop\Resume Analyser\resume"
nlp = spacy.load("en_core_web_sm")

def list_files(folder_path):
    files = []
    for filename in os.listdir(folder_path):
        if os.path.isfile(os.path.join(folder_path, filename)):
            files.append(os.path.join(folder_path, filename))  # Append the file path to the list of files
    return files

# Function to read resumes from a folder
def read_resumes(folder_path):
    resumes = []
    file_paths = list_files(folder_path)
    for file_path in file_paths:
        if file_path.endswith('.pdf'):
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)  # Replace PdfFileReader with PdfReader
                resume_text = ''
                for page_num in range(len(reader.pages)):
                    page = reader.pages[page_num]
                    resume_text += page.extract_text()
                if resume_text.strip():
                    resumes.append(resume_text)
        else:
            with open(file_path, 'r') as file:
                resume_text = file.read()
                if resume_text.strip():
                    resumes.append(resume_text)
    return resumes


# Function for text preprocessing
def preprocess_text(text):
    doc = nlp(text.lower())
    tokens = [token.text for token in doc]
    return ' '.join(tokens)

# Prompt the user to input the folder containing resumes
# folder_path = input("Enter the path to the folder containing resumes: ")

# Read resumes from the folder
resumes = read_resumes(folder_path)
print("Number of resumes found:", len(resumes))  # Debugging print statement

# Preprocess resumes
preprocessed_resumes = [preprocess_text(resume) for resume in resumes if resume.strip()]  # Only preprocess non-empty resumes
print("Number of preprocessed resumes:", len(preprocessed_resumes))  # Debugging print statement

if preprocessed_resumes:
   # Prompt the user to input the job description
    job_description = input("Enter the job description: ")

    # Preprocess the job description
    preprocessed_job_description = preprocess_text(job_description)

    # TF-IDF vectorization
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(preprocessed_resumes)
    job_vector = vectorizer.transform([preprocessed_job_description])

    # Calculate cosine similarity
    similarities = cosine_similarity(X, job_vector)

    # Sort resumes by similarity
    sorted_resumes = [(resumes[i], similarities[i][0]) for i in range(len(resumes))]
    sorted_resumes.sort(key=lambda x: x[1], reverse=True)

    # Print sorted resumes
    for resume, similarity in sorted_resumes:
        print("Similarity:", similarity)
        print(resume)
        print()

else:
    print("No valid resumes found in the folder.")
