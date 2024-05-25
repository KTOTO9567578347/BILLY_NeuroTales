import httpcore
setattr(httpcore, 'SyncHTTPTransport', 'AsyncHTTPProxy')

from googletrans import Translator, constants
from pprint import pprint
import nltk
import re
import nltk
import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from diffusers import StableDiffusionPipeline


def preprocess_text(text):
    # Tokenize the text into sentences and words
    sentences = sent_tokenize(text)
    words = [word_tokenize(sentence) for sentence in sentences]

    # Convert words to lowercase
    words = [[word.lower() for word in sentence] for sentence in words]

    # Remove stopwords and punctuation
    stop_words = set(stopwords.words('english'))
    words = [
        [word for word in sentence if word.isalnum() and word not in stop_words]
        for sentence in words
    ]

    return words

def sentence_similarity(sentence1, sentence2):
    # Convert sentences to CountVectorizer format
    sentence1 = ' '.join(sentence1)
    sentence2 = ' '.join(sentence2)
    vectorizer = CountVectorizer().fit_transform([sentence1, sentence2])
    vectors = vectorizer.toarray()

    # Calculate cosine similarity between sentences
    similarity = cosine_similarity(vectors)[0][1]
    return similarity

def text_rank(text, num_sentences=3):
    sentences = preprocess_text(text)

    # Create a similarity matrix between sentences
    similarity_matrix = np.zeros((len(sentences), len(sentences)))
    for i in range(len(sentences)):
        for j in range(len(sentences)):
            if i != j:
                similarity_matrix[i][j] = sentence_similarity(sentences[i], sentences[j])

    # Apply PageRank algorithm to get sentence scores
    scores = np.array([np.sum(similarity_matrix[i]) for i in range(len(sentences))])

    # Get the top sentences based on the scores
    ranked_sentences_indices = np.argsort(-scores)[:num_sentences]
    summary = [sentences[i] for i in sorted(ranked_sentences_indices)]
    summary = [' '.join(sentence) for sentence in summary]

    return ' '.join(summary)

def refresh_image_from_prompt(txt, name):
    translator = Translator()

    translation = translator.translate(txt)

    dat = translation.text

    res = text_rank(dat)

    pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", revision="fp16", torch_dtype=torch.float16)

    pipe.to('cuda')
    prompt = res

    image = pipe(prompt+"richly detailed Illustration").images[0]
    image.save(fr"image{name}.png")