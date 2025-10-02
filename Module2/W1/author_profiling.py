import streamlit as st
from nltk import PorterStemmer, WordNetLemmatizer, word_tokenize, download
import string
from collections import Counter
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Download NLTK resources once
download('punkt')
download('wordnet')

class NgramProfile:
    """Generate n-gram profiles of a given text for author/style comparison."""

    def __init__(self):
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
    
    def get_profile(self, text, n=2):
        """Return n-gram frequency profile for the given text."""
        tokens = self._process_text(text)
        n_gram_list = self._ngrams(tokens, n)
        return Counter(n_gram_list)
    
    def _process_text(self, text):
        """Tokenize, lowercase, lemmatize, and stem text."""
        tokens = word_tokenize(text.lower())
        processed_tokens = []
        for token in tokens:
            if token in string.punctuation:
                continue
            lemma = self.lemmatizer.lemmatize(token)
            stem = self.stemmer.stem(lemma)
            processed_tokens.append(stem)
        return processed_tokens
    
    def _ngrams(self, tokens, n=2):
        """Generate n-grams from a list of tokens."""
        if n <= 0:
            raise ValueError("n must be a positive integer")
        if len(tokens) < n:
            return []
        return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]

class ProfileComparator:
    """Compare two texts based on cosine similarity of their n-gram profiles."""

    def __init__(self, n=2):
        self.profile_generator = NgramProfile()
        self.n = n
    
    def compare_texts(self, text1, text2):
        """Return cosine similarity between two text profiles."""
        if not text1 or not text2:
            raise ValueError("Both text inputs are required")
        
        profile1 = self.profile_generator.get_profile(text1, n=self.n)
        profile2 = self.profile_generator.get_profile(text2, n=self.n)
        
        all_ngrams = set(profile1.keys()).union(set(profile2.keys()))
        
        vec1 = [profile1.get(ngram, 0) for ngram in all_ngrams]
        vec2 = [profile2.get(ngram, 0) for ngram in all_ngrams]
        
        vec1 = np.array(vec1).reshape(1, -1)
        vec2 = np.array(vec2).reshape(1, -1)
        
        return cosine_similarity(vec1, vec2)[0][0]


# Streamlit App
st.set_page_config(page_title="Author Profiling - Text Comparison", layout="centered")
st.title("Author Profiling - Text Comparison")

st.markdown("""
This application checks whether two uploaded texts are stylistically similar, 
based on **n-gram text analysis**.
""")

st.header("Step 1: Upload two text files to compare")

file1 = st.file_uploader("Choose first file (.txt)", type="txt", key="file1")
file2 = st.file_uploader("Choose second file (.txt)", type="txt", key="file2")

text1, text2 = None, None
if file1 is not None:
    text1 = file1.read().decode("utf-8")
    st.text_area("Content of first file:", text1, height=200)
if file2 is not None:
    text2 = file2.read().decode("utf-8")
    st.text_area("Content of second file:", text2, height=200)

st.header("Step 2: Similarity Result")

if st.button("Compare"):
    comparator = ProfileComparator(n=2)  # using bigrams
    if text1 and text2:
        similarity = comparator.compare_texts(text1, text2)
        st.write(f"**Cosine Similarity:** {similarity:.4f}")

        if similarity > 0.7:  # Example threshold
            st.success("Conclusion: The two texts are likely from the same author.")
        else:
            st.warning("Conclusion: The two texts are likely from different authors.")
    else:
        st.error("Please upload both files before comparing.")
