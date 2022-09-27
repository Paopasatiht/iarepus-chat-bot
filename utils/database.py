from crate import client

class DataStore():

    def __init__(self):

        self.connection = client.connect("localhost:4200", timeout = 3)
        self.cursor = self.connection.cursor()


    def push_to_database(self, intents, question, answer, probability, question_vec, status="pass"):

        self.cursor.execute(
                "INSERT INTO demo (intents,question, answer, probability, status, quevec) VALUES (?,?,?,?,?,?)", [intents,question, answer,probability, status, question_vec]
                )
