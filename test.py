import keyboard
import sounddevice as sd
import soundfile as sf
import numpy as np
from queue import Queue
import threading

# 1. Заранее загружаем звук в память
try:
    audio_data, sample_rate = sf.read("sound/ekh.wav", dtype='float32')
except Exception as e:
    print(f"Ошибка загрузки файла: {e}")
    exit()

# 2. Настройка низколатентного аудиопотока
sd.default.samplerate = sample_rate
sd.default.channels = audio_data.shape[1] if len(audio_data.shape) > 1 else 1
sd.default.latency = 'low'  # Режим низкой задержки
sd.default.blocksize = 256  # Уменьшаем размер блока

# 3. Очередь для воспроизведения
audio_queue = Queue()

def audio_worker():
    """Фоновый поток для воспроизведения"""
    with sd.OutputStream() as stream:
        while True:
            data = audio_queue.get()
            if data is None:  # Сигнал остановки
                break
            stream.write(data)

# Запускаем фоновый поток
audio_thread = threading.Thread(target=audio_worker, daemon=True)
audio_thread.start()

# 4. Обработчик клавиш с минимальной задержкой
def on_key_press(event):
    if event.name == 'a':
        audio_queue.put(audio_data.copy())  # Копируем данные для thread safety
    print(f'Нажата: {event.name}')

keyboard.on_press(on_key_press)
print("Готово. Нажмите 'a' для звука, 'Esc' для выхода")
keyboard.wait('esc')

# Очистка
audio_queue.put(None)
audio_thread.join()
