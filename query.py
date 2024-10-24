QUERY_ECG_TABLE = """
CREATE TABLE ECG (
    id INT AUTO_INCREMENT PRIMARY KEY,
    file_path VARCHAR(255) NOT NULL,
    upload_date DATE NOT NULL,
    bpm INT,
    other_health_info TEXT
);
"""


QUERY_DATA_TABLE = """
CREATE TABLE DIABETE (
    id INT AUTO_INCREMENT PRIMARY KEY,
    pregnancies INT NOT NULL,
    glucose INT NOT NULL,
    bloodPressure INT NOT NULL,
    skinThickness INT NOT NULL,
    insulin INT NOT NULL,
    BMI DECIMAL(5,1) NOT NULL,
    diabetesPedigreeFunction DECIMAL(5,3) NOT NULL,
    age INT NOT NULL,
    outcome TINYINT(1) NOT NULL
);
"""


QUERY_DELETE_TABLE ="""
DROP TABLE {table_name}
"""