from sklearn.neighbors import KNeighborsClassifier 
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis 
from sklearn.model_selection import GridSearchCV
from logger_config import my_logger


class DCIS_classification_model:
    """Contains 4 classification algorithms to use for DCIS risk classification
    
    Parameters:
    model_type: string representing the algorithm to use; can be either svm, lda, forest, or knn"""
    def __init__(self, model_type: str):
        self.model = None
        self.model_type = model_type.lower()
        self.use_grid_search = False
        
        if model_type == "svm":
            self.model = SVC()
        elif model_type == "knn":
            self.model = KNeighborsClassifier()
        elif model_type == "forest":
            self.model = RandomForestClassifier()
        elif model_type == "lda":
            self.model = LinearDiscriminantAnalysis()
        else:
            raise ValueError("Incompatable model type")
        
    def get_param_grid(self):
        """Gets a parameter grid to be used for Grid Search CV"""

        if self.model_type == "svm":
            return {'C': [0.1, 1, 10],'kernel': ['linear', 'rbf']}
        elif self.model_type == "knn":
            return {'n_neighbors': [3, 5, 7]}
        elif self.model_type == "forest":
            return {'n_estimators': [50, 100, 300]}
        elif self.model_type == "lda":
            return {'shrinkage': [None, 'auto']}
        else:
            raise ValueError("Incompatible model type")
    
        
    def fit(self, X_train, y_train, use_grid_search):
        """Trains a model with or without grid search cv
        
        Parameters:
        - X_train: training dataset of features
        - y_train: training dataset of labels
        - use_grid_search: boolean that determines whether or not to use GridSearchCV for training"""
        self.use_grid_search = use_grid_search

        if self.use_grid_search:
            param_grid = self.get_param_grid()
            grid_search = GridSearchCV(self.model, param_grid, cv=5, scoring='accuracy')
            grid_search.fit(X_train, y_train)
            self.model = grid_search
            my_logger.info(f"Best parameters for {self.model_type.upper()}: {grid_search.best_params_}")

        else:
            self.model.fit(X_train, y_train)
        
    
    def predict(self, X_test):
        """Returns predicted classes of the trained model"""
        return self.model.predict(X_test)
    
    def predict_prob(self, X_test):
        """Returns predicted classes probabilities of the trained model"""
        return self.model.predict_proba(X_test)
