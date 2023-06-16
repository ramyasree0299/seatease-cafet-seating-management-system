from flask import Flask

app = Flask(__name__)


@app.route('/checkSeatStatus')
def sendSeatStatus():
   response = [{
        "table_name" : "choix_1",
        "total_seats" : 4,
        "occupied" : 0,
        "station": "choix"
    },
    {
            "table_name" : "choix_2",
            "total_seats" : 4,
            "occupied" : 3,
            "station": "choix"
    }
]
   return response
    
if __name__ == "__main__":
    app.run(host="0.0.0.0",port=5000,debug=True)
    