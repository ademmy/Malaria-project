from flask import Flask
from flask_cors import CORS
from flask import request
from input_pipeline import input_pipeline
import json


app = Flask(__name__)
CORS(app)
@app.route("/", methods = ['POST'])
def hello():
	# req = b'{"test":[{"Joint Pain":"0","Body Pain":"0","Convulsion":"1","Headache":"1","Palor":"1","Vomiting":"1","Bitter Mouth":"0","Fever":"0","Catarah":"1","Anemia":"1","Cough":"1","Dizzyness":"0","Sore Throat":"1","Weakness":"1","Appetite":"0","Restlessness":"0","Unconciousness":"1","PVC":"0"}]}'
	req = request.data
	
	my_json = req.decode('utf8')
	print(my_json)
	data = json.loads(my_json)
	json_data = json.dumps(data, indent=4,sort_keys=True)
	print(json_data)
	resp = predict(json_data)
	return resp[0]





def predict(res):
	print (res)
	#reader= json_reader.json_reader()
	#dict_features= json_reader.json_parser(res)
	resData = json.loads(res)["test"]
	dic_features = []
	for i in resData:
                dic_features = i
                print(i)
	df=input_pipeline.create_dataframe(dic_features)
	prediction= input_pipeline.model(df)

	data= {}
	data['key'] = prediction
	print(data)
	#json_data = json.dumps(data)
	return prediction
if __name__ == "__main__":
	app.run()
