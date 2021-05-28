from flask import Flask, request, jsonify
from flask_basicauth import BasicAuth
from textblob import TextBlob
#import pandas as pd
#from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle
import os

modelo = pickle.load(open('models/modelo.sav','rb'))

#df = pd.read_csv('casas.csv')
colunas = ['tamanho','ano','garagem']


#x = df.drop('preco', axis=1)
##y = df['preco']

#x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
#modelo = LinearRegression()
#modelo.fit(x_train,y_train)

app = Flask(__name__)
app.config['BASIC_AUTH_USERNAME'] = os.environ.get('BASIC_AUTH_USERNAME')
app.config['BASIC_AUTH_PASSWORD'] = os.environ.get('BASIC_AUTH_PASSWORD')

basic_auth = BasicAuth(app)

@app. route('/')
def home():
    return "Minha Primeira API."

@app.route('/sentimento/<frase>')
@basic_auth.required
def sentimento(frase):
    tb = TextBlob(frase)
    tb_en = tb.translate(to='en')
    polaridade = tb_en.sentiment.polarity
    return "polaridade: {}".format(polaridade)

@app.route('/cotacao/', methods=['POST'])
@basic_auth.required
def cotacao():
    dados = request.get_json()
    dados_input = [dados[col] for col in colunas]
    preco = modelo.predict([dados_input])
    return jsonify(preco=preco[0])

if __name__== '__main__':
    app.run(debug = True, host = '0.0.0.0')
