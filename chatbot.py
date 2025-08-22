import nltk
import string

def preprocess(text_file):
    """
    Reads a text file, splits it into sentences, and preprocesses each sentence.
    """
    try:
        with open(text_file, 'r', encoding='utf-8') as file:
            text = file.read()
    except FileNotFoundError:
        return "Error: Text file not found."
    
    # Sentence tokenization
    sentences = nltk.sent_tokenize(text)
    
    # Preprocessing loop
    preprocessed_sentences = []
    for sentence in sentences:
        # Convert to lowercase
        sentence = sentence.lower()
        # Remove punctuation
        sentence = sentence.translate(str.maketrans('', '', string.punctuation))
        # Add to the new list
        preprocessed_sentences.append(sentence)
        
    return preprocessed_sentences
    from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def get_most_relevant_sentence(query, corpus):
    """
    Finds the most relevant sentence in the corpus for a given query.
    """
    # Append the query to the corpus for vectorization
    corpus_with_query = corpus + [query]
    
    # Initialize TF-IDF Vectorizer
    vectorizer = TfidfVectorizer()
    
    # Fit and transform the corpus and query
    tfidf_matrix = vectorizer.fit_transform(corpus_with_query)
    
    # Calculate cosine similarity between the query and all sentences
    # The last row of the matrix is the query vector
    query_vector = tfidf_matrix[-1]
    cosine_scores = cosine_similarity(query_vector, tfidf_matrix[:-1])
    
    # Find the index of the highest score
    most_relevant_index = cosine_scores.argmax()
    
    # Return the corresponding sentence from the original corpus
    return corpus[most_relevant_index]
    def chatbot(user_query, corpus):
        """
        Main chatbot logic. Handles user queries and returns a response.
        """
        user_query_lower = user_query.lower()
    
    # Handle greetings
    greetings = ["hello", "hi", "hey"]
    if any(greeting in user_query_lower for greeting in greetings):
        return "Hello! How can I help you today?"
        
    # Handle other queries
    if user_query_lower:
        relevant_sentence = get_most_relevant_sentence(user_query_lower, corpus)
        
        # Check if the sentence is relevant enough (optional but good practice)
        # You can add a threshold on the cosine similarity score if needed
        # For this simple case, we'll assume the result is always relevant
        return relevant_sentence
        
    return "I am sorry, I do not understand."
    import streamlit as st

def main():
    st.title("My Simple Chatbot 🤖")
    st.write("Hello! I can answer questions about AI history. Ask me anything!")

    # Load and preprocess the data
    corpus_sentences = preprocess("ai_history.txt")

    # Get user input
    user_query = st.text_input("You:", key="user_query")
    
    # Display chatbot response
    if user_query:
        if "bye" in user_query.lower():
            st.write("Bot: Goodbye! Have a great day!")
        else:
            response = chatbot(user_query, corpus_sentences)
            st.write(f"Bot: {response}")

if __name__ == "__main__":
    main()