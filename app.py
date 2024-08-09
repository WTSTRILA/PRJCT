from flask import Flask

app = Flask(__name__)

a = 5

@app.route('/')
def hello():
    return "module1"

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000)
