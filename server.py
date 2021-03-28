from argparse import ArgumentParser
from flask import Flask, request, jsonify
import json
import numpy

# python3 server.py -p 2501 -r de -m bert-base-cased
# python3 server.py -p 2502 -r en -m bert-base-german-cased

# uberspace web backend set bert.qiekub.org/en --http --port 2501
# uberspace web backend set bert.qiekub.org/de --http --port 2502

parser = ArgumentParser()
parser.add_argument('-p', '--port',
                    dest='port',
                    help='choose the port',
                    default='2500',
                    metavar='PORT'
                    )
parser.add_argument('-r', '--route',
                    dest='route',
                    help='choose the route',
                    default='/',
                    metavar='ROUTE'
                    )
parser.add_argument('-m', '--model',
                    dest='modelname',
                    help='choose the model',
                    default='bert-base-cased',
                    metavar='MODEL',
                    # choices=['bert-base-cased', 'bert-base-german-cased']
                    )
args = parser.parse_args()

port = args.port
route = '/'+args.route
modelname = args.modelname

if modelname == 'distilbert-base-uncased':
    from transformers import DistilBertTokenizer, DistilBertModel
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBertModel.from_pretrained('distilbert-base-uncased')
else:
    from transformers import BertTokenizer, BertModel
    tokenizer = BertTokenizer.from_pretrained(modelname)
    model = BertModel.from_pretrained(modelname)

if modelname == 'bert-base-cased' or modelname == 'distilbert-base-uncased':
    title = 'English Bert'
elif modelname == 'bert-base-german-cased':
    title = 'German Bert'
else:
    title = 'Bert'

print('Started '+title+'.')

class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """

    def default(self, obj):
        if isinstance(obj, (numpy.int_, numpy.intc, numpy.intp, numpy.int8,
                            numpy.int16, numpy.int32, numpy.int64, numpy.uint8,
                            numpy.uint16, numpy.uint32, numpy.uint64)):
            return int(obj)
        elif isinstance(obj, (numpy.float_, numpy.float16, numpy.float32,
                              numpy.float64)):
            return float(obj)
        elif isinstance(obj, (numpy.ndarray,)):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

app = Flask(__name__)

@app.route(route, methods=['GET', 'POST'])
def index():

    if request.method == 'GET':
        return '<title>'+title+'</title><h1>'+title+'</h1>Example: <code>curl -X POST https://bert.qiekub.org'+route+' -F \'text=How old are you?\'</code>'

    if request.method == 'POST':
        text = request.form.get('text', default='', type=str)
        result = []

        if text != '':
            encoded_input = tokenizer(text, return_tensors='pt')
            output = model(**encoded_input)

            if modelname == 'distilbert-base-uncased':
                result = output.last_hidden_state.detach().numpy()[0]
            else:
                result = output.pooler_output.detach().numpy()[0]

        return json.dumps(result.size, cls=NumpyEncoder), 200


# print('Gonna listen on 0.0.0.0:'+port)

if __name__ == '__main__':
    from waitress import serve
    serve(app, host="0.0.0.0", port=port)  # B=2 E=5 R=18 (T=20)
    # app.run(use_reloader=True, debug=True, host='0.0.0.0', port=port)


# curl -X POST 0.0.0.0:2518 -F 'lang=de' -F 'text=Wie alt bist du?'
