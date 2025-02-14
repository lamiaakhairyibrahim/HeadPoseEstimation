from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import joblib
import pandas as pd
class models:
    def __init__(self , featuers ,labels ):
        self.featuers  =  featuers 
        self.labels = labels 
        self.svr_parameters  = {'kernel':['linear', 'poly', 'rbf', 'sigmoid'],'C':[0.01,0.1,1,10,100]}
    def split(self):

        X_train,X_val,y_train,y_val = train_test_split(self.featuers,self.labels,test_size = 0.2,random_state = 20)
        y_train_pitch = y_train['pitch']
        y_train_yaw = y_train['yaw']
        y_train_roll = y_train['roll']
        y_val_pitch = y_val['pitch']
        y_val_yaw = y_val['yaw']
        y_val_roll = y_val['roll']
        return X_train,X_val , y_train_pitch , y_train_yaw , y_train_roll , y_val_pitch , y_val_yaw ,  y_val_roll
    
    def svcpitch(self):
        X_train,X_val , y_train_pitch , y_train_yaw , y_train_roll , y_val_pitch , y_val_yaw ,  y_val_roll = self.split()
        #svr_parameters = {'kernel':['linear', 'poly', 'rbf', 'sigmoid'],'C':[0.01,0.1,1,10,100]}
        # grid search pitch
        svr = SVR()
        svr_gs_pitch = GridSearchCV(estimator = svr,param_grid = self.svr_parameters )
        svr_gs_pitch.fit(X_train ,  y_train_pitch)
        svr_winner_pitch = svr_gs_pitch.best_estimator_
        print("Pitch Winner Model: ",svr_winner_pitch)
        print("Train Error: ",mean_absolute_error(svr_winner_pitch.predict(X_train),y_train_pitch))
        print("Validation Error: ",mean_absolute_error(svr_winner_pitch.predict(X_val),y_val_pitch))
        self.save_model(svr_winner_pitch , "svr_winner_pitch.pkl")
        return svr_winner_pitch
        
    def svcyaw(self):
        # grid search yaw
        X_train,X_val , y_train_pitch , y_train_yaw , y_train_roll , y_val_pitch , y_val_yaw ,  y_val_roll = self.split()
        svr = SVR()
        svr_gs_yaw = GridSearchCV(estimator = svr,param_grid = self.svr_parameters )
        svr_gs_yaw.fit(X_train, y_train_yaw)
        svr_winner_yaw = svr_gs_yaw.best_estimator_
        print("Yaw Winner Model: ",svr_winner_yaw)
        print("Train Error: ",mean_absolute_error(svr_winner_yaw.predict(X_train),y_train_yaw))
        print("Validation Error: ",mean_absolute_error(svr_winner_yaw.predict(X_val),y_val_yaw))
        self.save_model(svr_winner_yaw, "svr_winner_yaw.pkl")
        return svr_winner_yaw
    
    def svcroll(self):
        # grid search roll
        X_train,X_val , y_train_pitch , y_train_yaw , y_train_roll , y_val_pitch , y_val_yaw ,  y_val_roll = self.split()
        svr = SVR()
        svr_gs_roll = GridSearchCV(estimator = svr,param_grid =self.svr_parameters )
        svr_gs_roll.fit(X_train, y_train_roll)
        svr_winner_roll = svr_gs_roll.best_estimator_
        print("Yaw Winner Model: ",svr_winner_roll)
        print("Train Error: ",mean_absolute_error(svr_winner_roll.predict(X_train),y_train_roll))
        print("Validation Error: ",mean_absolute_error(svr_winner_roll.predict(X_val),y_val_roll))
        self.save_model(svr_winner_roll, "svr_winner_roll.pkl")
        return svr_winner_roll

    def save_model(self , model , filename ):
        joblib.dump(model , filename)
        print(f"model saved as {filename}")
    @staticmethod
    def load_model(filename):
        return joblib.load(filename)

    