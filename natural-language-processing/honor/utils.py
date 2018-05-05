import nltk
import pickle
import re
import numpy as np
import pandas as pd

nltk.download('stopwords')
from nltk.corpus import stopwords

# Paths for all resources for the bot.
RESOURCE_PATH = {
    'INTENT_RECOGNIZER': 'intent_recognizer.pkl',
    'TAG_CLASSIFIER': 'tag_classifier.pkl',
    'TFIDF_VECTORIZER': 'tfidf_vectorizer.pkl',
    'THREAD_EMBEDDINGS_FOLDER': 'thread_embeddings_by_tags',
    'WORD_EMBEDDINGS': 'word_embeddings.tsv',
}


def text_prepare(text):
    """Performs tokenization and simple preprocessing."""
    
    replace_by_space_re = re.compile('[/(){}\[\]\|@,;]')
    bad_symbols_re = re.compile('[^0-9a-z #+_]')
    stopwords_set = set(stopwords.words('english'))

    text = text.lower()
    text = replace_by_space_re.sub(' ', text)
    text = bad_symbols_re.sub('', text)
    text = ' '.join([x for x in text.split() if x and x not in stopwords_set])

    return text.strip()


def load_embeddings(embeddings_path):
	"""Loads pre-trained word embeddings from tsv file.

	Args:
	embeddings_path - path to the embeddings file.

	Returns:
	embeddings - dict mapping words to vectors;
	embeddings_dim - dimension of the vectors.
	"""

	# Hint: you have already implemented a similar routine in the 3rd assignment.
	# Note that here you also need to know the dimension of the loaded embeddings.

	embeds = pd.read_csv(embeddings_path,sep="\t",header=None)
	vals=embeds.iloc[:,1:].values
	index=embeds.iloc[:,0].values
	embeddings= {i:j for i,j in zip(index,vals)}
	return embeddings,vals.shape[1]

       
def question_to_vec(question, embeddings, dim):
    """Transforms a string to an embedding by averaging word embeddings."""
    
    # Hint: you have already implemented exactly this function in the 3rd assignment.

    if question == "":
        return np.zeros(dim)
    t = np.array([embeddings[i]
                  for i in question.split() if i in embeddings.keys()])
    if len(t) == 0:
        return np.zeros(dim)

    return(t.mean(axis=0))


def unpickle_file(filename):
    """Returns the result of unpickling the file content."""
    with open(filename, 'rb') as f:
        return pickle.load(f)
def sentence_to_ids(sentence, word2id, padded_len):
    """ Converts a sequence of symbols to a padded sequence of their ids.
    
      sentence: a string, input/output sequence of symbols.
      word2id: a dict, a mapping from original symbols to ids.
      padded_len: an integer, a desirable length of the sequence.

      result: a tuple of (a list of ids, an actual length of sentence).
    """
    
    sent_ids = [word2id[i] for i in sentence]
    sent_len = len(sent_ids[:padded_len-1])+1
    sent_ids = sent_ids[:padded_len-1]+[word2id["$"]]+[word2id["#"]]*(padded_len-len(sent_ids)-1)
    
    return (sent_ids, sent_len)
def ids_to_sentence(ids, id2word):
    """ Converts a sequence of ids to a sequence of symbols.
    
          ids: a list, indices for the padded sequence.
          id2word:  a dict, a mapping from ids to original symbols.

          result: a list of symbols.
    """
 
    return [id2word[i] for i in ids] 
def batch_to_ids(sentences, word2id, max_len):
    """Prepares batches of indices. 
    
       Sequences are padded to match the longest sequence in the batch,
       if it's longer than max_len, then max_len is used instead.

        sentences: a list of strings, original sequences.
        word2id: a dict, a mapping from original symbols to ids.
        max_len: an integer, max len of sequences allowed.

        result: a list of lists of ids, a list of actual lengths.
    """
    
    max_len_in_batch = min(max(len(s) for s in sentences) + 1, max_len)
    batch_ids, batch_ids_len = [], []
    for sentence in sentences:
        ids, ids_len = sentence_to_ids(sentence, word2id, max_len_in_batch)
        batch_ids.append(ids)
        batch_ids_len.append(ids_len)
    return batch_ids, batch_ids_len
def generate_batches(samples, batch_size=64):
    X, Y = [], []
    for i, (x, y) in enumerate(samples, 1):
        X.append(x)
        Y.append(y)
        if i % batch_size == 0:
            yield X, Y
            X, Y = [], []
    if X and Y:
        yield X, Y
def reply(question,word2id,max_len,model,id2word):

    ids, ids_len = sentence_to_ids(question,word2id,padded_len=max_len)
    ids=np.array(ids).reshape(1,len(ids))

    ids_len=np.array(ids_len).reshape(1)
    predictions = model.predict_for_batch(session, ids, ids_len)
    return "".join(ids_to_sentence(predictions[0], id2word)).replace("$","").capitalize()
