from flask import Flask
from flask import request
from flask_cors import CORS, cross_origin
from flask import jsonify
from PIL import Image
from base64 import decodestring, b64decode
app = Flask(__name__)
CORS(app)

@app.route('/home', methods=['GET'])
def greeting():
    return 'Hello, World!'

@app.route('/request', methods=['POST'])
def answer():
    #But really, convert request.data to a PIL image and give it to pytorch
    #then get the letter pytorch thinks is right
    header, encoded = request.data.split(b",", 1)
    dat = b64decode(encoded)

    open("image.jpg", "wb").write(dat)
    #For now, return a placeholder
    response = jsonify({'letter':'a'})
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response
    

if __name__ == '__main__':
    app.run(debug=True)