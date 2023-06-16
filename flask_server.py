from flask import Flask
from database_operations import *

app = Flask(__name__)

@app.route('/checkSeatStatus')
def sendSeatStatus():
  response = showFetchAPIData()
  return response

def showFetchAPIData():
    conn_cursor.execute("""SELECT * from table_occupancy""")
    table_occupancy = conn_cursor.fetchall()
    print(table_occupancy)
    results = []
    for result in table_occupancy:
        result_flask_results =  dict()
        result_flask_results["table_name"] = result[0]
        result_flask_results["total_seats"] = result[1]
        result_flask_results["occupied"] = check_result(result[0],result[2])
        result_flask_results["station"] = result[0].split("_")[0]
        results.append(result_flask_results)
#   print(results)
    return results

def check_result(table_name,occupied):
  if table_name == "Choix_1": 
    return 0
  else: return occupied
  
if __name__ == "__main__":
    app.run(host="0.0.0.0",port=5000,debug=True)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
  