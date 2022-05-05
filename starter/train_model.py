# Script to train machine learning model.
from sklearn.model_selection import train_test_split

# Add the necessary imports for the starter code.
from .ml.data import process_data, get_clean_data
from .ml.model import train_model, compute_model_metrics, inference
from sklearn.preprocessing import StandardScaler


def train(tuning=True):
    """
    Train and save a model
    :return:
    """
    # Add code to load in the data.
    data = get_clean_data()

    # Optional enhancement, use K-fold cross validation instead of a train-test split.
    train, test = train_test_split(data, test_size=0.20)

    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]
    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=cat_features, label="salary", training=True
    )
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)

    # Process the test data with the process_data function.
    X_test, y_test, _, _ = process_data(test, categorical_features=cat_features, label="salary",
                                        training=False, encoder=encoder, lb=lb)
    X_test = scaler.transform(X_test)

    # Train and save a model.
    model = train_model(X_train, y_train, tuning)
    model.save_model()

    y_pred = inference(model, X_test)
    # precision, recall, and F1
    evaluation = compute_model_metrics(y_test, y_pred)

    return evaluation






