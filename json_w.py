import json

class json_read:

    def json_reader(json_file):

        with open(json_file, 'r') as file:
            data= json.load(file)
        return data


    def json_parser(data):
        for values in data["test"]:
            data_points= values
        return values
    
