# Disaster Response Pipeline Project

### Table of Contents

1. [Installation](#installation)
2. [Project Motivation](#motivation)
3. [File Descriptions](#files)
4. [Screenshots](#screenshots)
5. [Acknowledgements and appeal](#licensing)

## Installation <a name="installation"></a>

The project can be easily built in an Anaconda environment using the given `requirements.txt` and with `conda create --name <env> --file requirements.txt`.

### Dependencies

The project uses the following libraries:
- Python 3.9.7
- Machine Learning: NumPy, SciPy, Pandas, Sciki-Learn
- Natural Language Processing: NLTK
- SQLlite Database: SQLalchemy
- Web Development: Flask
- Visualization: Plotly

### Instructions:

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://localhost:3001/

## Project Motivation<a name="motivation"></a>

This project focuses on message analysis in response to real-world disasters. I built a Natural Language Processing tool that categorizes the messages into disaster types. The labeled dataset was provided by https://appen.com/ (previously Figure Eight).

The project is divided in the following steps:

- ETL Pipeline to extract data from source, process and clean the data and save them in an SQLite database
- Machine Learning Pipeline to train a NLP model using the Adaboost classifier to categorize messages
- Web dashboard to visualize the dataset and perform real-time predictions on new messages.

## Limitations

Not all categories in the dataset are represented equally. Some categories, e.g. `child_alone`, `shops` or `tools` only contain a handful of categorized training messages. This might lead to misclassification of certain messages due to underfitting.

## File Descriptions <a name="files"></a>

- model: 
    - `train_classifier.py`: The NLP pipeline producing a Adaboost model `classifier.pkl`
- data:
    - `process_data.py`: The ETL pipeline to read, clean two excel-files and load them into a joind SQLite database `DisasterResponse.db`
- app:
    - `run.py`: Process to locally start the Flask dashboard

## Screenshots <a name="screenshots"></a>

1. Entering a message
![image](https://user-images.githubusercontent.com/8439378/158881037-45c3c6d1-c441-4882-99ba-dc11012e2fea.png)

2. Obtain the predicted message
![image](https://user-images.githubusercontent.com/8439378/158881177-406120f8-7da0-4276-b3f0-48514e577f08.png)

3. Visualization of training data
![image](https://user-images.githubusercontent.com/8439378/158881647-f9aca699-1afa-4f56-8c58-0461978f03cd.png)
![image](https://user-images.githubusercontent.com/8439378/158881734-fd6ded3e-5c0a-4316-a437-a48192395406.png)
![image](https://user-images.githubusercontent.com/8439378/158881865-1fc71788-b686-472c-b229-8ee19bce3a66.png)

## Acknowledgements and appeal<a name="licensing"></a>

- I want to thank https://appen.com/ for providing the dataset..
- If you find yourself in a dangerous situation, natural disaster or similar, please reach out to dedicated help centers. For disaster assistance in Germany, reach out to https://www.bmi.bund.de/EN/topics/civil-protection/bbk/bbk-node.html.
- License: MIT
