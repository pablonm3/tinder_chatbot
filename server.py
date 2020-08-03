from app import chat
from flask import Flask, request
app = Flask(__name__)

@app.route("/")
def handler():
    event = request.json
    print(event)
    return chat(event.get('user_id', None), event.get('chat_id', None), event.get('msg', None))

if __name__ == "__main__":
    app.run()