
import mysql.connector


class DataBaseConnection(object): 

    def __init__(self):
        self.db = mysql.connector.connect(
            host="localhost",
            user="root",
            password="password",  # Replace with your actual MySQL root password
            database="your_database_name"  # Optional: Specify a default database if needed
        )



    # Making a table and all and all of the other databases to make them work very very properly right


class DB_Service(object): 
    def __init__(self): 
        pass 





# QUERIES 
"""CREATE TABLE file_info (
    id INT AUTO_INCREMENT PRIMARY KEY,
    file_path VARCHAR(255) NOT NULL,
    upload_date DATE NOT NULL,
    bpm INT,
    other_health_info TEXT
);"""




