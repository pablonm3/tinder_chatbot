import spacy
import nltk
from nltk.corpus import stopwords
import unidecode
import re
from simpletransformers.language_representation import RepresentationModel
from sklearn.metrics.pairwise import cosine_similarity

nlp = spacy.load("es", disable=['parser', 'tagger', 'ner'])
nltk.download('stopwords')
stops = stopwords.words("spanish")


reply_id_to_text = {
    "opener": "Robemos un banco!",
    "a2_1": "Perfecto, hoy a las 22 en el Bar Verne Club ahí discutiremos el plan, nos vemos en la puerta, me vas a reconocer por que tendré un pasamontañas rosa!",
    "a2_shittest": "Vamos, el plan es el siguiente, vos en bikini y yo en sunga y con pistolas de agua 💦, vamos a ser el Hit 💣💣💣",
    "default": "Bancame que termino de entrenar y te cuento mejor, tenes cara de ansiosa!"
}
msg_text_to_reply_id = {
    "Ya": "a2_1",
    "si cuando": "a2_1",
    "Dale Ya": "a2_1",
    "A que hora": "a2_1",
    "dale, nos vemos en": "a2_1",
    "eh?": "a2_shittest",
    "?": "a2_shittest",
    "wtf?": "a2_shittest",
    "pff": "a2_shittest",
    "U+1F926": "a2_shittest"# facepalm emoji unicode, validate this
}

msg_text_to_sen_embedding = {}

def remove_accented_chars(text):
    """remove accented characters from text, e.g. café"""
    text = unidecode.unidecode(text)
    return text


def remove_URL(text):
    url = re.compile(r'https?://\S+|www\.\S+')
    return url.sub(r'', str(text))


def remove_emoji(text):
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)


def remove_html(text):
    html = re.compile(r'<.*?>')
    return html.sub(r'', text)


def remove_punctuation(text):
    table = str.maketrans('', '', string.punctuation)
    return text.translate(table)


# FIXME: for lemmatizing, normalizing and removing stop words we tokenize the text and then join all the tokens using Python String.join, when doing this info like some punctuation marks are dropped which reduces model's accuracy
# If you find a way to over come this problem please let me know in the comments!
def text_normalizer(comment, lemmatize, lowercase, remove_stopwords, remove_accents,
                    normalize_URL, normalize_emoji, normalize_html, normalize_punctuation):
    if lowercase:
        comment = comment.lower()
    if (remove_accents):
        comment = remove_accented_chars(comment)
    if (normalize_URL):
        comment = remove_URL(comment)
    if (normalize_emoji):
        comment = remove_emoji(comment)
    if (normalize_html):
        comment = remove_html(comment)
    if (normalize_punctuation):
        comment = remove_punctuation(comment)
    if (remove_stopwords):
        comment = nlp(comment)
        words = [];
        for token in comment:
            if not remove_stopwords or (remove_stopwords and token.text not in stops):
                words.append(token.text)
        comment = " ".join(words)
    if (lemmatize):
        comment = nlp(comment)
        comment = " ".join(word.lemma_.strip() for word in comment)
    return comment


sentences = ["Example sentence 1", "Example sentence 2"]
model = RepresentationModel(
    model_type="bert",
    model_name="dccuchile/bert-base-spanish-wwm-uncased",
    use_cuda=False
)


def generate_msg_embedding(msg):
    prepropped_msg = text_normalizer(msg, lemmatize=False,
                                     lowercase=False,
                                     remove_stopwords=False,
                                     remove_accents=True,
                                     normalize_URL=True,
                                     normalize_emoji=True,
                                     normalize_html=True,
                                     normalize_punctuation=False)

    return model.encode_sentences([prepropped_msg], combine_strategy="mean")




for text in msg_text_to_reply_id:
    embedding = generate_msg_embedding(text)
    msg_text_to_sen_embedding[text] = embedding

def find_closest_response(message_embedding):
    closest_text = None;
    highest_cos_sim = 0
    min_threshold = 0.6;
    for text, embedding in msg_text_to_sen_embedding.items():
        cos_sim = cosine_similarity(embedding, message_embedding)[0][0]
        if(cos_sim > min_threshold and cos_sim > highest_cos_sim):
            highest_cos_sim = cos_sim
            closest_text = text;
    if(closest_text):
        reply_id = msg_text_to_reply_id[closest_text]
    else:
        reply_id = "default"
    return reply_id_to_text[reply_id], highest_cos_sim

def chat(user_id=None, chat_id=None, msg=None):
    if(msg == None):
        # create new chat
        reply = reply_id_to_text["opener"]
        # add reply to chat history
        return reply
    message_embedding = generate_msg_embedding(msg)
    closest_response, _ = find_closest_response(message_embedding)
    return closest_response



