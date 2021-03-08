
import pandas as pd
from flask import Flask, jsonify, request
import pickle

# load model
model = pickle.load(open('model.pkl', 'rb'))

# create app
app = Flask(__name__)

# Routes
@app.route('/', methods=['POST'])

def predict():
	# get the data
	data = request.get_json(force=True)

	# convert data to dataframe
	data.update((x, [y]) for x, y in data.items())
	data_df = pd.DataFrame.from_dict(data)

	# make predictions
	result = model.predict(data_df)

	# send the output
	output = {'results':int(result[0])}

	# return the output
	return jsonify(results=output)

if __name__ == '__main__':
	app.run(port=8000, debug=True)