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
                "Gradient Boosting":GradientBoostingClassifier(),
                "KNeighbors Classifier":KNeighborsClassifier(),
                "Catboost Classifer":CatBoostClassifier(verbose=False),
                "Adaboost Classifier":AdaBoostClassifier()
            }
            params={
                "Random Forest":{
                    'n_estimators':[8,16,32,64,128,256]
                },
                "Decision Tree":{
                    'criterion':["gini", "entropy", "log_loss"]
                },
                "Gradient Boosting":{
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "KNeighbors Classifier":{
                    'n_neighbors':[5,7,9,11]
                },
                "Catboost Classifer":{
                    "depth":[6,8,10],
                    "iterations":[30,50,100],
                    "learning_rate":[0.01,0.5,0.1]
                },
                "Adaboost Classifier":{
                    'learning_rate':[0.01,0.05,0.1],
                    'n_estimators':[8,16,32,64,128,256]
                }
            }
            model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models,param=params)
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

