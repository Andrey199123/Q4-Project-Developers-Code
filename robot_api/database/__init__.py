import sqlite3


class Database:
    def __init__(self):
        # Create a connection to the sqlite3 database, create the table structure, and then close connection
        connection = sqlite3.Connection("database.db")
        cursor = connection.cursor()

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS accounts(
                ID INTEGER NOT NULL,
                Username TEXT,
                Password TEXT,
                PRIMARY KEY (ID)
            )
            """
        )

        connection.commit()
        connection.close()

    def check_for_account(self, username: str) -> bool:
        # Create a connection to the sqlite3 database, check if the user exists, then close the connection
        connection = sqlite3.Connection("database.db")
        cursor = connection.cursor()
        result = cursor.execute(
            "SELECT * FROM accounts WHERE username=?", (username,)
        ).fetchone()
        connection.commit()
        connection.close()
        return result is not None

    def login(self, username: str, password: str) -> bool:
        # Create a connection to the sqlite3 database, find the password associated with the username provided,
        # and then check the provided password against the one provided
        connection = sqlite3.Connection("database.db")
        cursor = connection.cursor()
        if not self.check_for_account(username):
            print(
                f"Attempt to login although account {username} does not exist."
            )
            return False

        account_password = cursor.execute(
            "SELECT password FROM accounts WHERE username=?", (username,)
        ).fetchone()[0]

        if account_password is None:
            print(f"Account {username} does not have an associated password.")

        connection.commit()
        connection.close()
        return account_password == password

    def register(self, username: str, password: str):
        # Register a new user into the database,
        # adds a username and password with a unique identifier
        connection = sqlite3.Connection("database.db")
        cursor = connection.cursor()
        user_id = cursor.execute("SELECT MAX(ID) FROM accounts").fetchone()[0]

        if user_id is None:
            user_id = 0
        else:
            user_id += 1

        # encrypted_password = hashlib.scrypt(password.encode(), n=32, r=8, p=1, salt=os.urandom(32))

        # self.__cursor.execute("INSERT INTO accounts VALUES(?, ?, ?)", (user_id, username, encrypted_password))
        cursor.execute(
            "INSERT INTO accounts VALUES(?, ?, ?)",
            (user_id, username, password),
        )
        connection.commit()
        connection.close()
