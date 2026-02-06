import numpy as np
import pandas as pd


emails = pd.read_csv('data/emails.csv')
print('\nData Head:')
print(emails.head())

def process_email(text):
    '''
    Golden function for processing text.
    :param text: String to process
    :return: List of lowercase-words
    '''
    text = text.lower()
    return list(set(text.split()))

emails['words'] = emails['text'].apply(process_email)
print('\nPre-processed Data:')
print(emails.head())

num_emails = len(emails)
num_spam = sum(emails['spam']) # Sum of spam

print("\nNumber of emails:", num_emails)
print("Number of spam emails:", num_spam)

# Prior probability
print("\nPrior probability of spam (spam emails / total emails):", num_spam/num_emails)

def count_words(emails):
    '''
    Bag-of-Words approach. Create spam and ham counters for each word.
    '''
    model = {}
    for index, email in emails.iterrows(): # Iterate over all emails
        for word in email['words']: # Iterate over words in the email
            if word not in model: # First occurrence of the word
                model[word] = {'spam': 1, 'ham': 1} # Initialize counters (1 instead of 0 = Laplace smoothing)
            if word in model:
                if email['spam']:
                    model[word]['spam'] += 1
                else:
                    model[word]['ham'] += 1
    return model

model = count_words(emails)

print('\nSpam and ham counters for selected words (1 means 0):')
print(f'Lottery: {model['lottery']}')
print(f'Sale: {model['sale']}')
print(f'Mum: {model['mum']}')

def predict_bayes(word):
    word = word.lower() # input pre-processing
    num_spam_with_word = model[word]['spam'] # dictionary. E.g. mum:'spam' = 1
    num_ham_with_word = model[word]['ham'] # dictionary. E.g. mum:'ham' = 4
    # P(spam|word) = P(word|spam)/(P(word|spam)+(P(word|ham))
    return 1.0*num_spam_with_word/(num_spam_with_word + num_ham_with_word)


print('\nBayes:')
print(f'Word Lottery: {predict_bayes('lottery')}')
print(f'Word Sale: {predict_bayes('sale')}')


def predict_naive_bayes(email):
    total = len(emails)
    num_spam = sum(emails['spam']) # P(spam)
    num_ham = total - num_spam # P(ham)
    email = email.lower() # preprocessing
    words = set(email.split()) # preprocessing
    spams = [1.0] # Initial probability
    hams = [1.0] # Initial probability

    log_spams = np.log(num_spam)
    log_hams = np.log(num_ham)
    max_log = max(log_spams, log_hams)

    # Before:
    # Score = P(spam) * P(spam|word1) * P(spam|word2) * ... = UNDERFLOW
    # After:
    # LogScore = logP(spam) + logP(spam|word1) + logP(spam|word2) + ...

    for word in words: # Iterates over each unique word
        if word in model:
            log_spams += np.log(model[word]['spam'] / num_spam) # E.g. mum:'spam' = 1
            log_hams += np.log(model[word]['ham'] / num_ham) # E.g. mum:'spam' = 4

    prob_spam = np.exp(log_spams - max_log) # Posterior probability of spam
    prob_ham  = np.exp(log_hams  - max_log) # Posterior probability of ham
    return prob_spam / (prob_spam + prob_ham)

print('\nEnter your words:')
input_words = input()
print(f'Probability that the email {input_words} is spam: {predict_naive_bayes(input_words)}')