from telegram.ext import Application, CommandHandler, MessageHandler, filters


from bot_telegram import *


def main():
    print('Starting bot...')
    app = Application.builder().token(TOKEN).build()

    # Commands
    app.add_handler(CommandHandler('start', start_command))
    # app.add_handler(CallbackQueryHandler(button_click))
    app.add_handler(CommandHandler('help', help_command))
    # app.add_handler(CommandHandler('custom', custom_command))
    app.add_handler(CommandHandler('chat', handle_chat_command))
    app.add_handler(CommandHandler('email', handle_email_command))

    # Messages
    app.add_handler(MessageHandler(filters.TEXT, handle_message))

    # Error
    app.add_error_handler(error)

    # Polling
    print('polling...')
    app.run_polling(poll_interval=3)



    
if __name__ == '__main__':
  main()  
    
    