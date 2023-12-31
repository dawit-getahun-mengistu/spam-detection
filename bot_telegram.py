#  setting up and training our ML models before starting our bot 
 ##################################################################


# importing  our machine learning models
from NaiveBayes import NaiveBayes
from LogisticRegression import LogisticRegression
    
# importing training variables for the email dataset using bag of words feature extraction
from bag_of_words import vectorize as vectorize_bow, vectorize_list as vectorize_list_bow
from bag_of_words import training_set as email_training_set, testing_set as email_testing_set
from bag_of_words import training_labels as email_training_labels, testing_labels as email_testing_labels

from bag_of_words import LABELS

# importing training variables for the sms dataset using bigram feature extraction
from bigram import vectorize as vectorize_bigram, vectorize_list as vectorize_list_bigram
from bigram import training_set as sms_training_set, testing_set as sms_testing_set
from bigram import training_labels as sms_training_labels, testing_labels as sms_testing_labels




# setting up Naive Bayes machine learning model for email dataset
X_train_email = vectorize_list_bow(email_training_set)
X_test_email = vectorize_list_bow(email_testing_set)

y_train_email = [LABELS.index(label) for label in email_training_labels]
y_test_email = [LABELS.index(label) for label in email_testing_labels]

# making sure training happens on the entire dataset
email_train = []
email_train.extend(X_train_email)
email_train.extend(X_test_email)
email_train_labels = []
email_train_labels.extend(y_train_email)
email_train_labels.extend(y_test_email)

nb = NaiveBayes(learning_rate=0.1, alpha=0.1)


# setting up Logistic Regression machine learning model for sms dataset
X_train_sms = vectorize_list_bigram(sms_training_set)
X_test_sms = vectorize_list_bigram(sms_testing_set)

y_train_sms = [LABELS.index(label) for label in sms_training_labels]
y_test_sms = [LABELS.index(label) for label in sms_testing_labels]

sms_train = []
sms_train.extend(X_train_sms)
sms_train.extend(X_test_sms)
sms_train_labels = []
sms_train_labels.extend(y_train_sms)
sms_train_labels.extend(y_test_sms)

logReg = LogisticRegression(learning_rate=0.1)
logReg.fit(sms_train, sms_train_labels)
predict = logReg.predict([vectorize_bigram('text')])



# boolean values to check if training was success
nb_successfully_trained = False
logReg_successfully_trained = False


# training both models
try:    
    nb.fit(email_train, email_train_labels)
    nb_successfully_trained = True
    
    logReg.fit(sms_train, sms_train_labels)
    logReg_successfully_trained = True
except:
    nb_successfully_trained = False
    logReg_successfully_trained = False
     




#############################################################################################################
##########################      SETTING UP THE TELEGRAM BOT      ##########################################
#############################################################################################################

# importing important modules to setup telegram bot
from typing import final
from telegram import Update
from telegram.ext import ContextTypes

# importing dotenv for handling environment variables
from dotenv import dotenv_values



config = dotenv_values('.env')
TOKEN: final = config['TOKEN']
BOT_USERNAME: final = config['BOT_USERNAME']


# boolean to know which dataset is requested
use_email_dataset = False
use_sms_dataset = True

# switch models
def switch_models():
    use_email_dataset = not use_email_dataset
    use_sms_dataset = not use_sms_dataset

# Commands
menu_options = ['text chat spam detector', 'email spam detector']    
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    

    await update.message.reply_text(
        'Hello, I am a spam detector bot. I can help you protect yourself from phishing and other malicious activities on Telegram. If you suspect a message or a channel is spam, forward it to me and I will analyze it and give you a report. You can also report spam directly to Telegram by using the @SpamBot contact. Stay safe and enjoy Telegram!'
        )
    
    
    
async def button_click(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global use_email_dataset
    global use_sms_dataset
    
    query = update.callback_query
    data = query.data

    print(data)
    if data == menu_options[0]:
        use_email_dataset = True
        use_sms_dataset = False
        # update.message.reply_text('Email spam detector ready')
        
    if data == menu_options[1]:
        use_email_dataset = False
        use_sms_dataset = True
        # update.message.reply_text('chat text spam detector ready')


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Welcome! \n This bot is designed to help you identify and handle spam messages in your Telegram chats. It utilizes advanced machine learning techniques to analyze the content of incoming messages and determine their spam probability. Here's how you can make the most out of this bot: \n  Forward or send any suspicious message to this bot, and it will analyze the content to determine its spam likelihood. \n Please note the following: \n \t - While the spam detection algorithm is highly accurate, it may occasionally produce false positives or false negatives. It's always good to exercise caution and make your own judgment. \n \t - This bot continuously learns from user feedback to improve its spam detection capabilities. Your input is valuable! You can provide feedback on false positives or false negatives to help enhance the bot's accuracy. \n \t - To ensure privacy and data security, this bot does not store any user messages or personal information. The content analysis is performed in real-time and discarded after processing.\n '/chat' - handle text messages through chat \n '/email' - handle email messages "
)
    
    

async def handle_chat_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global use_email_dataset
    global use_sms_dataset
    use_email_dataset = False
    use_sms_dataset = True
    await update.message.reply_text('Chat Spam Detector is Ready!')

async def handle_email_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global use_email_dataset
    global use_sms_dataset
    use_email_dataset = True
    use_sms_dataset = False
    await update.message.reply_text('email Spam Detector is Ready!')


# Responses
def handle_response(text: str) -> str:
    if text.strip() == '':
        return 'I do not understand what you wrote'
    
    if nb_successfully_trained:
        vector = [vectorize_bow(text)]
        if use_email_dataset:
            prediction = nb.predict(vector)[0]
            if prediction == 1:
                return 'This email message is more likely to be a spam message'
            elif prediction == 0:
                return 'This email message is less likely to be a spam message'
            
    if logReg_successfully_trained:
        vector = [vectorize_bigram(text)]
        if use_sms_dataset:
            prediction = logReg.predict(vector)[0]
            if prediction == 1:
                return 'This text message is more likely to be a spam message'
            elif prediction == 0:
                return 'This text message is less likely to be a spam message'
            
    else:
        return 'not trained.'



async def handle_message(update: Update, constext: ContextTypes.DEFAULT_TYPE):
    message_type: str = update.message.chat.type
    text: str = update.message.text
    
    if text.strip() == '':
        return ''
    
    
    # if text in menu_options: return
    if text not in menu_options:
        if message_type == 'group':
            if BOT_USERNAME in text:
                text: str = text.replace(BOT_USERNAME, '').strip()
            response: str = handle_response(text)
            if response.strip() == 'spam':
                await update.message.delete()
                
            else:
                return
        else:
            response: str = handle_response(text)
            await update.message.reply_text(response)    
        print('Bot:', response)
    
    
    
async def error(update: Update, context: ContextTypes.DEFAULT_TYPE):
    print(f'Update {update} caused error {context.error}')
    
    
if __name__ == '__main__':
    print(logReg.predict([vectorize_bigram('hello how are you')]))
    print(nb.predict([vectorize_bow('hello how are you')]))