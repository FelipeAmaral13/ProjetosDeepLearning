import os
import sys
import time
import magic
import telepot
from telepot.loop import MessageLoop

from yolo_predict import ObjectDetector

detector = ObjectDetector()


def get_file_type(file_path):
    try:
        return magic.from_file(file_path, mime=True)
    except Exception as e:
        return f"Error: {e}"

def handle_message(msg):
    content_type, chat_type, chat_id = telepot.glance(msg)

    if content_type == 'document':
        file_id = msg['document']['file_id']
        file_path = file_id

        try:
            bot.download_file(file_id, file_path)
            file_mime_type = get_file_type(file_path)

            if file_mime_type in ['image/png', 'image/jpeg']:
                detections = detector.detect(file_path)
                labels = [detect[5]['class_name'] for detect in detections]
                label_counts = {label: labels.count(label) for label in set(labels)}
                mensagem = "*Classes detectadas na imagem:* \n"
                for label, count in label_counts.items():
                    mensagem += f"{label} = {count}\n"

                detector.visualize(file_path, detections, save_path="annotated_image.jpg")
                with open("annotated_image.jpg", "rb") as image_file:
                    bot.sendPhoto(chat_id, image_file)
            else:
                mensagem = "*Apenas imagens PNG ou JPG são suportadas.*"
            bot.sendMessage(chat_id, mensagem, parse_mode='Markdown')
            os.remove(file_path)
            os.remove('annotated_image.jpg')
        except Exception as e:
            bot.sendMessage(chat_id, f"Error: {e}", parse_mode='Markdown')


if __name__ == '__main__':
    token = sys.argv[1]
    bot = telepot.Bot(token)
    MessageLoop(bot, handle_message).run_as_thread()

    while True:
        time.sleep(2)
