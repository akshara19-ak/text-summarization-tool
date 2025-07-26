import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer
import numpy as np
import networkx as nx
import re
nltk.download('punkt')
nltk.download('stopwords')

class TextSummarizer:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()
    
    def preprocess_text(self, text):
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        words = word_tokenize(text.lower())                
        words = [self.stemmer.stem(word) for word in words if word not in self.stop_words]
        return words
    
    def build_similarity_matrix(self, sentences):       
        similarity_matrix = np.zeros((len(sentences), len(sentences)))      
        for i in range(len(sentences)):
            for j in range(len(sentences)):
                if i == j:  
                    continue
                words_i = self.preprocess_text(sentences[i])
                words_j = self.preprocess_text(sentences[j])
                intersection = len(set(words_i).intersection(set(words_j)))
                union = len(set(words_i).union(set(words_j)))
                similarity_matrix[i][j] = intersection / union if union != 0 else 0
                
        return similarity_matrix
    
    def summarize(self, text, num_sentences=3):
    
        sentences = sent_tokenize(text)
        
        
        if len(sentences) <= num_sentences:
            return text
        
    
        similarity_matrix = self.build_similarity_matrix(sentences)
        
        sentence_graph = nx.from_numpy_array(similarity_matrix)
        
        scores = nx.pagerank(sentence_graph)
        
    
        ranked_sentences = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)
        
      
        top_sentences = [s for score, s in ranked_sentences[:num_sentences]]
        
        
        summary_sentences = sorted(top_sentences, key=lambda s: sentences.index(s))
        
     
        summary = ' '.join(summary_sentences)
        
        return summary

def main():
    print("Text Summarization Tool")
    print("-----------------------")
    
    
    sample_text = """
    Natural language processing (NLP) is a subfield of linguistics, computer science, 
    and artificial intelligence concerned with the interactions between computers and human language. 
    It focuses on how to program computers to process and analyze large amounts of natural language data. 
    The result is a computer capable of understanding the contents of documents, including the contextual 
    nuances of the language within them. The technology can then accurately extract information and insights 
    contained in the documents as well as categorize and organize the documents themselves. 
    Challenges in natural language processing frequently involve speech recognition, natural language 
    understanding, and natural language generation. Modern NLP algorithms are based on machine learning, 
    especially statistical machine learning. The paradigm of machine learning is different from that of 
    most prior attempts at language processing. Prior implementations often involved direct hand-coding 
    of large sets of rules.
    """
    
    summarizer = TextSummarizer()
    
    while True:
        print("\nOptions:")
        print("1. Summarize sample text")
        print("2. Enter your own text")
        print("3. Read text from file")
        print("4. Exit")
        
        choice = input("Enter your choice (1-4): ")
        
        if choice == '1':
            text = sample_text
        elif choice == '2':
            print("Enter your text (press Enter twice to finish):")
            lines = []
            while True:
                line = input()
                if line.strip() == '':
                    if len(lines) > 0:
                        break
                    else:
                        continue
                lines.append(line)
            text = '\n'.join(lines)
        elif choice == '3':
            filename = input("Enter filename: ")
            try:
                with open(filename, 'r', encoding='utf-8') as file:
                    text = file.read()
            except FileNotFoundError:
                print("File not found. Please try again.")
                continue
        elif choice == '4':
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please try again.")
            continue
        
        num_sentences = int(input("How many sentences should the summary contain? "))
        
        summary = summarizer.summarize(text, num_sentences)
        
        print("\nOriginal Text Length:", len(text), "characters")
        print("Summary Length:", len(summary), "characters")
        print("\nSummary:")
        print(summary)

if __name__ == "__main__":
    main()

    