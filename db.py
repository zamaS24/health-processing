
"""
    NOTES: 
        - We have a cursor and the methods ar internat ott this cursor
"""


import mysql.connector







if __name__ == '__main__': 

    db = mysql.connector.connect(
    host="localhost",
    user="root",
    password="password",  # Replace with your actual MySQL root password
    database="health_db"  # Optional: Specify a default database if needed
)

    cursor = db.cursor()

    query = 'show tables'
    cursor.execute(query)

    for x in cursor:
        print(type(x)) 
        print(x)




