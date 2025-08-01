import numpy as np
import matplotlib.pyplot as plt

from models import DCIS_classification_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score



def evaluate_model():
    # path = 'data/transformed_data/merged_df.csv'
    # merged_df = pd.read_csv(path)
    # merged_df = merged_df.iloc[:, 1:]

    # merged_df.dropna(inplace = True)
    # print(merged_df.shape)

    X = np.random.rand(3000, 10)
    y = np.random.randint(0, 3, size=(3000, 1)).ravel()
    model = DCIS_classification_model("forest")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42, shuffle=True)
    model.fit(X_train, y_train, use_grid_search=True)
    y_pred = model.predict(X_test)

    print(f"Displaying classification report for {model.model_type.upper()}: \n")
    print(classification_report(y_pred=y_pred, y_true=y_test))
    print(f"The overall accuracy of {model.model_type.upper()} is {accuracy_score(y_pred=y_pred,y_true=y_test)}")


if __name__ == "__main__":
    evaluate_model()
