# Обработчик подключения нового клиента к общей комнате
import socketio
from aiohttp import web
from openai import OpenAI
from image_generator import refresh_image_from_prompt
import os




client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")
imgnum = 1
last_emit = None

# Создаем объект сервера
sio = socketio.AsyncServer(cors_allowed_origins="*", ping_timeout = 10000000000000) # Разрешаем кросс-доменные запросы, и делаем большой таймаут для 
app = web.Application()
sio.attach(app)

@sio.event
async def connect(sid, environ):
	print(f'Клиент {sid} подключен к общей комнате')
	# Присоединяем клиента к общей комнате
	await sio.enter_room(sid, 'common_room')

# Обработчик отключения клиента от общей комнаты
@sio.event
async def disconnect(sid):
	print(f'Клиент {sid} отключен от общей комнаты')
	#Покидаем общую комнату при отключении
	await sio.leave_room(sid, 'common_room')

# Обработчик нового сообщения от клиента
@sio.event
async def message(sid, data):
	print(f'Получено сообщение от {sid}: {data}')
	global imgnum
	# Отправляем сообщение всем клиентам в общей комнате, кроме отправителя
	if data['author'] == 'Пользователь':
		prompt_text = data['text']
		
		resp = client.chat.completions.create(
            model="local-model", # this field is currently unused
            messages=[
            {"role": "system", "content": "Говори как древнерусский добрый бард-сказочник."},
            {"role": "user", "content": 'Используй только русский язык.' + prompt_text}
            ],
            temperature=0.7,
            )
		ans_text = resp.choices[0].message.content
		ans_text.replace('\n', '')
		ans_data = {'text': ans_text, 'author': 'Сказочник'}

		print("Ответ пользователю:", ans_data, '\n')

		refresh_image_from_prompt(ans_text, imgnum)
		imgnum+=1
		last_emit = ans_data
		await sio.emit('message',ans_data, room='common_room')
		print('Emit succsessfull!')
	#await sio.emit('message', data, room='common_room', skip_sid=sid)
	
# Запуск сервера
if __name__ == '__main__':
	web.run_app(app, port=5000)