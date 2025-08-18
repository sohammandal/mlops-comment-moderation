import pandas as pd
import nltk
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
import random
from multiprocessing import Pool
from tqdm import tqdm


# Download necessary NLTK data
def download_nltk_data():
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)
    try:
        nltk.data.find("corpora/wordnet")
    except LookupError:
        nltk.download("wordnet", quiet=True)
    try:
        nltk.data.find("taggers/averaged_perceptron_tagger")
    except LookupError:
        nltk.download("averaged_perceptron_tagger", quiet=True)


def get_synonyms(word):
    """
    Get synonyms of a word from WordNet.
    """
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name())
    if word in synonyms:
        synonyms.remove(word)
    return list(synonyms)


def synonym_replacement(text, n=1):
    """
    Replace n words in a text with their synonyms.
    """
    words = word_tokenize(text)
    new_words = words.copy()
    random_word_list = list(set([word for word in words if wordnet.synsets(word)]))
    random.shuffle(random_word_list)
    num_replaced = 0
    for random_word in random_word_list:
        synonyms = get_synonyms(random_word)
        if len(synonyms) >= 1:
            synonym = random.choice(synonyms)
            new_words = [synonym if word == random_word else word for word in new_words]
            num_replaced += 1
        if num_replaced >= n:
            break

    sentence = " ".join(new_words)
    return sentence


def augment_text(text):
    """
    Augments a single piece of text by replacing words with synonyms.
    """
    try:
        # Calculate number of words to replace (10% of words)
        num_words_to_replace = max(1, int(len(word_tokenize(text)) * 0.1))
        return synonym_replacement(text, n=num_words_to_replace)
    except Exception:
        return text


def parallel_augment(data, num_processes=12):
    """
    Augments a pandas Series in parallel.
    """
    with Pool(num_processes) as p:
        augmented_text = list(tqdm(p.imap(augment_text, data), total=len(data)))
    return augmented_text


if __name__ == "__main__":
    download_nltk_data()
    # Load the data
    df = pd.read_csv("comments_test.csv")

    # Select 70% of the data to augment
    sample = df.sample(frac=0.7, random_state=42)

    # Augment the selected data
    augmented_comments = parallel_augment(sample["comment_text"])

    # Create a new dataframe with the augmented data
    augmented_df = sample.copy()
    augmented_df["comment_text"] = augmented_comments

    # Create the v2 dataframe
    df_v2 = df.copy()
    df_v2.update(augmented_df)

    # Save the new data
    df_v2.to_csv("comments_test_v2.csv", index=False)

    print("Data augmentation complete. Saved to comments_test_v2.csv")
