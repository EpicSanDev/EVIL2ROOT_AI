from telegram import Bot
from config.config import TELEGRAM_TOKEN, TELEGRAM_CHAT_ID

class TelegramBot:
    def __init__(self):
        self.bot = Bot(token=TELEGRAM_TOKEN)

    def send_signal(self, signal):
        self.bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=signal)