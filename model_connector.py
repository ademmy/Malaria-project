from json_reader import json_reader
from preprocesing_module import MultiLabelEncoder
from input_pipeline import input_pipeline
import json


reader = json_reader.json_reader(request_file)
dict_features = json_reader.json_parser(reader)
df = input_pipeline.create_dataframe(dict_features)
mlt_data = input_pipeline.preprocessing_steps(df)
prediction = input_pipeline.model(mlt_data)

data = {}
data['key'] = prediction
json_data = json.dumps(data)
print(json_data)


