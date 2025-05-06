from flask import Flask, redirect, render_template, request, url_for

from robot_api.database import *

# Create Flask app
app = Flask(
    __name__,
    # static_url_path="/static",
    static_url_path="/",
    static_folder="site/static",
    template_folder="site/templates",
)

# Import routes from api subdirectory
import robot_api.api
import robot_api.camera

# Initialize database
database = Database()


# Default webpage route, redirects to /login route
@app.route("/")
def default():
    return redirect(url_for("login"))


# Login page route
@app.route("/login", methods=["POST", "GET"])
def login():
    # If the request is a POST request, check if the username exists, if not, create an account with the provided
    # username and password. If the username does exist, check if the provided password is valid. If not valid, redirect
    # the user back to the login page with an error message. If valid, redirect the user ot the home page.
    if request.method == "POST":
        if not database.check_for_account(request.form.get("username")):
            database.register(
                request.form.get("username"), request.form.get("password")
            )
            return render_template(
                "index.html", message="Successfully registered new account."
            )
        elif database.login(
            request.form.get("username"), request.form.get("password")
        ):
            return redirect(url_for("home"))
        else:
            return render_template(
                "index.html", error_message="Incorrect password."
            )

    # All GET requests will render the standard index.html page
    return render_template("index.html")


# Home page route
# @app.route("/home")
# def home():
#    return render_template("home.html")
