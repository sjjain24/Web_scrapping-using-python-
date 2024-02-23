from bs4 import BeautifulSoup
import requests
import pandas as pd
import os
import re
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('stopwords')

# Read input file
input_file = "Input.xlsx"
df = pd.read_excel(input_file)

# Function to extract article data from a given URL
def extract_article_data(url):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            title = soup.title.text.strip() if soup.title else "No Title Found"
            article_text = ""
            content_div = soup.find('div', class_='td-post-content tagdiv-type')
            if content_div:
                paragraphs = content_div.find_all(['p', 'li', 'h2','h3'])
                for paragraph in paragraphs:
                    article_text += paragraph.get_text() + "\n"
            else:
                article_text = ""
                content_div = soup.find('div', class_='tdb-block-inner td-fix-index')
                if content_div:
                    paragraphs = content_div.find_all(['p', 'li', 'h2','h3'])
                    for paragraph in paragraphs:
                        article_text += paragraph.get_text() + "\n"
                else:
                    print(f"Data extraction failed for URL: {url}. Content div not found.")
                    return None
                
            if not article_text:
                article_elements = soup.find_all(['p', 'li', 'h2','h3'])
                for element in article_elements:
                    article_text += element.text.strip() + '\n'
            
            return title, article_text
        else:
            print(f"Failed to retrieve the webpage. Status code: {response.status_code}")
            return None
    except Exception as e:
        error_message = f"Error occurred while fetching URL: {url}. Error: {str(e)}"
        print(error_message)
        # Save error message into a text file
        with open("error_log.txt", "a", encoding="utf-8") as error_file:
            error_file.write(error_message + "\n")
        return None


# Directory containing text files
text_files_directory = "C:\\Users\Shrut\OneDrive\Desktop\Assignment\Extracted_data"

# Iterate over each row in the DataFrame
for index, row in df.iterrows():
    url_id = row['URL_ID']
    url = row['URL']
    
    try:
        title, article_text = extract_article_data(url)
        
        # Save the extracted article text to a text file
        if title and article_text:
            filename = os.path.join(text_files_directory, f"{url_id}.txt")
            with open(filename, 'a', encoding='utf-8') as file:  # 'a' mode for append
                file.write(f"Title: {title}\n\n")
                file.write(article_text)
                print(f"Article text extracted and appended for URL_ID: {url_id}")
    except Exception as e:
        print(str(e))            

print("Extraction process completed.")

# List to store extracted information
data = []

# Iterate over each text file in the directory
for file_name in os.listdir(text_files_directory):
    if file_name.endswith('.txt'):
        file_path = os.path.join(text_files_directory, file_name)
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()  # Read the content of the file
            url_id = file_name.replace('.txt','')
            url = df[df['URL_ID'] == url_id]['URL'].values[0] # Extract URL_ID from file name
            data.append({'URL_ID': url_id, 'URL': url, 'text': content})  # Store file name and content

# Create a DataFrame from the extracted data
df = pd.DataFrame(data)

# Function to read master dictionary of positive and negative words
def read_master_dictionary(positive_file, negative_file):
    positive_words = set()
    with open(positive_file, 'r') as file:
        for line in file:
            positive_words.add(line.strip())
    negative_words = set()
    with open(negative_file, 'r') as file:
        for line in file:
            negative_words.add(line.strip())
    return positive_words, negative_words

# Function to read stopwords
def read_stopwords(stopwords_directory):
    stopwords_set = set()
    for file_name in os.listdir(stopwords_directory):
        if file_name.endswith('.txt'):
            file_path = os.path.join(stopwords_directory, file_name)
            with open(file_path, 'r') as file:
                for line in file:
                    stopwords_set.add(line.strip())
    return stopwords_set

# Define stopwords directory
stopwords_directory = r"C:\Users\Shrut\OneDrive\Desktop\Assignment\stopwords"
stopwords_set = read_stopwords(stopwords_directory)

# Function to analyze sentiment of the text
def analyze_sentiment(text, positive_words, negative_words, stopwords_set):
    cleaned_text = ' '.join([word for word in text.split() if word.lower() not in stopwords_set])
    tokens = word_tokenize(cleaned_text)
    positive_score = sum(1 for word in tokens if word.lower() in positive_words)
    negative_score = sum(1 for word in tokens if word.lower() in negative_words)
    polarity_score = (positive_score - negative_score) / ((positive_score + negative_score) + 0.000001)
    subjectivity_score = (positive_score + negative_score) / (len(tokens) + 0.000001)
    return positive_score, negative_score, polarity_score, subjectivity_score

# Function to analyze readability of the text
def analyze_readability(text):
    sentences = sent_tokenize(text)
    num_sentences = len(sentences)
    num_words = len(word_tokenize(text))
    avg_sentence_length = num_words / num_sentences
    words = [word.lower() for word in word_tokenize(text)]
    complex_words = [word for word in words if len(word) > 2]
    percentage_complex_words = len(complex_words) / num_words
    fog_index = 0.4 * (avg_sentence_length + percentage_complex_words)
    return avg_sentence_length, percentage_complex_words, fog_index

# Function to calculate average number of words per sentence
def average_words_per_sentence(text):
    num_sentences = len(sent_tokenize(text))
    num_words = len(word_tokenize(text))
    return num_words / num_sentences

# Function to count complex words
def count_complex_words(text):
    words = word_tokenize(text)
    complex_words = [word for word in words if len(word) > 2]
    return len(complex_words)

# Function to count syllables in a word
def syllable_count(word):
    vowels = 'aeiouy'
    word = word.lower()
    count = 0
    if word[0] in vowels:
        count += 1
    for index in range(1, len(word)):
        if word[index] in vowels and word[index - 1] not in vowels:
            count += 1
    if word.endswith('e'):
        count -= 1
    if word.endswith('le') and len(word) > 2 and word[-3] not in vowels:
        count += 1
    if count == 0:
        count += 1
    return count

# Function to count the number of words in text
def word_count(text):
    words = word_tokenize(text)  # Tokenize the text into words
    return len(words)  # Return the count of words

# Function to calculate syllables per word
def calculate_syllable_per_word(text):
    words = word_tokenize(text)
    syllable_count_list = [syllable_count(word) for word in words]
    return sum(syllable_count_list) / len(words)

# Function to count personal pronouns in text
def count_personal_pronouns(text):
    personal_pronouns = ["i", "we", "my", "ours", "us"]
    pattern = r'\b(?:' + '|'.join(personal_pronouns) + r')\b(?![a-z])'
    return len(re.findall(pattern, text.lower()))

# Function to calculate average word length
def average_word_length(text):
    words = word_tokenize(text)
    total_chars = sum(len(word) for word in words)
    return total_chars / len(words)

# Main function to perform text analysis
def main():
    # Read master dictionary
    positive_words, negative_words = read_master_dictionary('positive-words.txt', 'negative-words.txt')
    # List to store results
    results = []
    
    for index,row in df.iterrows():
        text = row['text']
        url_id = row['URL_ID']
        url = row['URL']

        # Analyze sentiment
        positive_score, negative_score, polarity_score, subjectivity_score = analyze_sentiment(text, positive_words, negative_words, stopwords_set)
        
        # Analyze readability
        avg_sentence_length, percentage_complex_words, fog_index = analyze_readability(text)
        
        # Calculate other metrics
        avg_words_per_sentence = average_words_per_sentence(text)
        complex_word_count = count_complex_words(text)
        total_word_count = word_count(text)
        syllable_per_word = calculate_syllable_per_word(text)
        personal_pronouns_count = count_personal_pronouns(text)
        avg_word_length = average_word_length(text)
        
        # Append results to list
        results.append({
            'URL_ID': url_id,
            'URL': url,
            'POSITIVE SCORE': positive_score,
            'NEGATIVE SCORE': negative_score,
            'POLARITY SCORE': polarity_score,
            'SUBJECTIVITY SCORE': subjectivity_score,
            'AVG SENTENCE LENGTH': avg_sentence_length,
            'PERCENTAGE OF COMPLEX WORDS': percentage_complex_words,
            'FOG INDEX': fog_index,
            'AVG NUMBER OF WORDS PER SENTENCE': avg_words_per_sentence,
            'COMPLEX WORD COUNT': complex_word_count,
            'WORD COUNT': total_word_count,
            'SYLLABLE PER WORD': syllable_per_word,
            'PERSONAL PRONOUNS': personal_pronouns_count,
            'AVG WORD LENGTH': avg_word_length
        })
    
    # Create DataFrame from results
    df_results = pd.DataFrame(results)
    
    # Save DataFrame to CSV
    df_results.to_csv('text_analysis_results.csv', index=False)

if __name__ == "__main__":
    main()
