from flask import Flask

app = Flask(__name__)


@app.route('/checkSeatStatus')
def sendSeatStatus():
   response = {
        "table_name" : "choix_1",
        "total_seats" : 6,
        "occupied" : 0,
        "station": "choix"
    }
   return response
    
if __name__ == "__main__":
    app.run(host="0.0.0.0",port=5000,debug=True)
    