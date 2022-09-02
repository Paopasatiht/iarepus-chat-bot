from pythainlp import sent_tokenize, word_tokenize
import numpy as np

def thai_tokenize(sentence):
    return word_tokenize(
                        sentence,
                        keep_whitespace=False
    )

def thai_bag_of_words(tokenized_sentence, words):
    """
    return bag of words array:
    1 for each known word that exists in the sentence, 0 otherwise
    example:
    sentence = ["hello", "how", "are", "you"]
    words = ["hi", "hello", "I", "you", "bye", "thank", "cool"]
    bog   = [  0 ,    1 ,    0 ,   1 ,    0 ,    0 ,      0]
    """
    # stem each word
    sentence_words = [word for word in tokenized_sentence]
    # initialize bag with 0 for each word
    bag = np.zeros(len(words), dtype=np.float32)
    for idx, w in enumerate(words):
        if w in sentence_words:
            bag[idx] = 1

    return bag

# test_text = 'superai มีขั้นตอนการสมัครเข้าร่วมโครงการอย่างไร?'
# test_text = thai_tokenize(test_text)
# # stem and lower each word
# ignore_words = ['?', '.', '!']
# all_words = [w for w in test_text if w not in ignore_words]
# all_words = sorted(set(all_words))
# print(all_words)