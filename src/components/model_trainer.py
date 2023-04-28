import sys
import os
from dataclasses import dataclass

from catboost import CatBoostClassifier
from sklearn.ensemble import AdaBoostClassifier,GradientBoostingClassifier,RandomForestClassifier
from sklearn.metrics import confusion_matrix,roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from src.exception import Custom_Exception
from src.logger import logging
from src.utils import evaluate_models, save_object

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info('Split Train and Test Data has started')
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],train_array[:,-1],test_array[:,:-1],test_array[:,-1]
            )
            models={
                "Random Forest":RandomForestClassifier(),
                "Decision Tree":DecisionTreeClassifier(),
                "Gradiant Boosting":GradientBoostingClassifier(),
                "KNeighbors Classifier":KNeighborsClassifier(),
                "Catboost Classifer":CatBoostClassifier(verbose=False),
                "Adaboost Classifier":AdaBoostClassifier()
            }
            model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models)
            best_model_score=max(sorted(model_report.values()))
            best_model_name=list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model=models[best_model_name]
            
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted=best_model.predict(X_test)
            auc_score=roc_auc_score(y_test,predicted)
            return auc_score

        except Exception as e:
            raise Custom_Exception(e,sys)

