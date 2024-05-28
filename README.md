# Сказочник БИЛЛИ
Репозиторий с кодом первого прототипа системы генерации сказок БИЛЛИ

#### На данный момент это не версия для полностью публичного использования, только демонстрационный вариант. В дальнейшем весь запуск будет автоматическим из исполняемого файла!

Для запуска:
- установите Cuda 11.8
- установите Microsoft C++ Build Tools
- в консоли создайте venv, далее всё запускайте из него
- в venv команды:
  - pip install openai
  - pip install python-socketio
  - pip install aiohttp
  - pip install googletrans==3.1.0a0
  - pip install httpx==0.27.0
  - pip install nltk
  - pip install numpy
  - pip install scikit-learn
  - pip install diffusers
  - pip install transformers
  - pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

- скачайте архив BILLY_user.zip отсюда, распакуйте.
- В директории BILLY_user откройте папку BILLY_user/app, запустите в вашем браузере файл index.html
- В LM studio:
    1) Откройте LMstudio
    2) В меню слева выберите LocalServer (значок "<-->")
    3) Нажмите Start Server (большая зелёная кнопка)
- запустите файл main.py из папки BILLY_user/app из консоли, находясь в вашем VENV
- Должна начаться загрузка данных, ждите. Не закрывайте консоль!
- Если вы видите надпись ======== Running on http://0.0.0.0:5000 ======== то всё готово

- теперь можно вписать запрос в чат и система примерно в течении минуты выдаст вам ответ
