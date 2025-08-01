from sklearn.neighbors import KNeighborsClassifier 
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis 
from sklearn.model_selection import GridSearchCV


class DCIS_classification_model:
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
        self.use_grid_search = use_grid_search

        if self.use_grid_search:
            param_grid = self.get_param_grid()
            grid_search = GridSearchCV(self.model, param_grid, cv=5, scoring='accuracy')
            grid_search.fit(X_train, y_train)
            self.model = grid_search
            print(f"Best parameters for {self.model_type.upper()}: {grid_search.best_params_}")

        else:
            self.model.fit(X_train, y_train)
        
    
    def predict(self, X_test):

        return self.model.predict(X_test)
    
