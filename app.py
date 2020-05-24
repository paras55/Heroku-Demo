from flask import Flask, render_template,request,jsonify
#from flask_socketio import SocketIO, emit,send,request,jsonify
#from flask_ngrok import run_with_ngrok
#from flask_cors import CORS, cross_origin
import random 
import time 
import numpy
#from PyDictionary import PyDictionary
import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()
import nltk
nltk.download('punkt')
import numpy
import tflearn
import tensorflow
import random
import json
import pickle
import os

#dictionary=PyDictionary()


app = Flask(__name__)

 def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
            
    return numpy.array(bag)

@app.route( '/chat' ,methods=['GET', 'POST'])
#@cross_origin(origin='*',headers=['access-control-allow-origin','Content-Type'])
def chat():
    #msg = request.get_json(force=True)
    content = request.json
    msg=content['text']
    #msg=request.get_json(force=True)
    

    with open("intents.json") as file:
        data = json.load(file)

    try:
        with open("data.pickle", "rb") as f:
            words, labels, training, output = pickle.load(f)
    except:
        words = []
        labels = []
        docs_x = []
        docs_y = []
    
        for intent in data["intents"]:
            for pattern in intent["patterns"]:
                wrds = nltk.word_tokenize(pattern)
                words.extend(wrds)
                docs_x.append(wrds)
                docs_y.append(intent["tag"])
    
            if intent["tag"] not in labels:
                labels.append(intent["tag"])
    
        words = [stemmer.stem(w.lower()) for w in words if w != "?"]
        words = sorted(list(set(words)))
    
        labels = sorted(labels)
    
        training = []
        output = []
    
        out_empty = [0 for _ in range(len(labels))]
    
        for x, doc in enumerate(docs_x):
            bag = []
    
            wrds = [stemmer.stem(w.lower()) for w in doc]
    
            for w in words:
                if w in wrds:
                    bag.append(1)
                else:
                    bag.append(0)
    
            output_row = out_empty[:]
            output_row[labels.index(docs_y[x])] = 1
    
            training.append(bag)
            output.append(output_row)
    
    
        training = numpy.array(training)
        output = numpy.array(output)
        with open("data.pickle", "wb") as f:
            pickle.dump((words, labels, training, output), f)

    tensorflow.reset_default_graph()
    
    net = tflearn.input_data(shape=[None, len(training[0])])
    net = tflearn.fully_connected(net, 8)
    net = tflearn.fully_connected(net, 8)
    net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
    net = tflearn.regression(net)
    
    model = tflearn.DNN(net)
    
    MODEL_NAME='model.tflearn'
    if os.path.exists(MODEL_NAME + ".meta"):
        model.load(MODEL_NAME)
    else:
        model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
        model.save(MODEL_NAME)
        
    results = model.predict([bag_of_words(msg, words)])
    results_index = numpy.argmax(results)
    tag = labels[results_index]

    for tg in data["intents"]:
            if tg['tag'] == tag:
                responses = tg['responses']

    return jsonify({'data' :random.choice(responses)})
   # return jsonify(random.choice(responses))



    

        
if __name__ == '__main__':
  app.run(debug=True)
  

