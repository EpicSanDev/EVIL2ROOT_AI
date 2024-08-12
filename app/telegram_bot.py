from telegram import Bot
from config.config import TELEGRAM_TOKEN, TELEGRAM_CHAT_ID

class TelegramBot:
    def __init__(self):
        self.bot = Bot(token=TELEGRAM_TOKEN)

    async def send_message(self, message):
        await self.bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=message)