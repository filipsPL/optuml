# Benchmark

Results of the benchmark with optuml for classification and regression using standard scikit-learn data sets.

```
===================================================================================================================
CLASSIFICATION BENCHMARK
===================================================================================================================

--- Iris (n=150, features=4, classes=3) ---
    Scoring: accuracy, CV folds: 5, timeout: 60s per run

  SVC... done (1.8s)
  KNeighborsClassifier... done (0.8s)
  RandomForestClassifier... done (30.3s)
  AdaBoostClassifier... done (38.9s)
  LogisticRegression... done (2.6s)
  DecisionTreeClassifier... done (0.7s)
  GaussianNB... done (0.6s)
  MLPClassifier... done (185.8s)

Algorithm                 Default CV   Default Test  Quick CV     Quick Test   Full CV      Full Test    Time(s)   
-------------------------------------------------------------------------------------------------------------------
SVC                       0.9750       0.9667        0.9833       1.0000       0.9833       1.0000       1.8       
KNeighborsClassifier      0.9750       1.0000        0.9750       0.9333       0.9750       0.9333       0.8       
RandomForestClassifier    0.9500       0.9000        0.9667       0.9667       0.9667       0.9667       30.3      
AdaBoostClassifier        0.9500       0.9333        0.9583       0.9333       0.9583       0.9333       38.9      
LogisticRegression        0.9667       0.9667        0.9750       1.0000       0.9750       1.0000       2.6       
DecisionTreeClassifier    0.9417       0.9333        0.9667       0.9333       0.9667       0.9333       0.7       
GaussianNB                0.9583       0.9667        0.9583       0.9667       0.9583       0.9667       0.6       
MLPClassifier             0.9667       1.0000        0.9833       0.9333       0.9833       0.9333       185.8     

--- Wine (n=178, features=13, classes=3) ---
    Scoring: accuracy, CV folds: 5, timeout: 60s per run

  SVC... done (60.2s)
  KNeighborsClassifier... done (0.8s)
  RandomForestClassifier... done (34.2s)
  AdaBoostClassifier... done (40.1s)
  LogisticRegression... done (34.7s)
  DecisionTreeClassifier... done (0.8s)
  GaussianNB... done (0.6s)
  MLPClassifier... done (762.0s)

Algorithm                 Default CV   Default Test  Quick CV     Quick Test   Full CV      Full Test    Time(s)   
-------------------------------------------------------------------------------------------------------------------
SVC                       0.6480       0.6944        0.9793       0.9444       0.9793       0.9444       60.2      
KNeighborsClassifier      0.7108       0.8056        0.7899       0.8611       0.7899       0.8611       0.8       
RandomForestClassifier    0.9862       1.0000        0.9862       1.0000       0.9862       1.0000       34.2      
AdaBoostClassifier        0.9648       0.9167        0.9581       0.9444       0.9650       0.9722       40.1      
LogisticRegression        0.9581       0.9722        0.9724       0.9722       0.9724       0.9722       34.7      
DecisionTreeClassifier    0.9163       0.9444        0.9227       0.9444       0.9232       0.9444       0.8       
GaussianNB                0.9719       0.9722        0.9791       0.9722       0.9791       0.9722       0.6       
MLPClassifier             0.4074       0.9167        0.9374       0.9722       0.9374       0.9722       762.0     

--- Breast Cancer (n=569, features=30, classes=2) ---
    Scoring: accuracy, CV folds: 5, timeout: 60s per run

  SVC... done (176.4s)
  KNeighborsClassifier... done (1.0s)
  RandomForestClassifier... done (36.8s)
  AdaBoostClassifier... done (73.9s)
  LogisticRegression... done (29.6s)
  DecisionTreeClassifier... done (1.2s)
  GaussianNB... done (0.6s)
  MLPClassifier... done (1113.5s)

Algorithm                 Default CV   Default Test  Quick CV     Quick Test   Full CV      Full Test    Time(s)   
-------------------------------------------------------------------------------------------------------------------
SVC                       0.9099       0.9298        0.9516       0.9649       0.9516       0.9649       176.4     
KNeighborsClassifier      0.9363       0.9123        0.9407       0.9474       0.9429       0.9474       1.0       
RandomForestClassifier    0.9538       0.9561        0.9560       0.9474       0.9560       0.9474       36.8      
AdaBoostClassifier        0.9692       0.9561        0.9780       0.9561       0.9780       0.9561       73.9      
LogisticRegression        0.9429       0.9649        0.9582       0.9737       0.9604       0.9737       29.6      
DecisionTreeClassifier    0.9099       0.9123        0.9319       0.9386       0.9341       0.9211       1.2       
GaussianNB                0.9363       0.9386        0.9385       0.9386       0.9407       0.9386       0.6       
MLPClassifier             0.9451       0.9561        0.9451       0.9035       0.9451       0.9035       1113.5    

===================================================================================================================
REGRESSION BENCHMARK
===================================================================================================================

--- Diabetes (n=442, features=10) ---
    Scoring: r2, CV folds: 5, timeout: 60s per run

  SVR... done (1.5s)
  KNeighborsRegressor... done (0.8s)
  RandomForestRegressor... done (38.4s)
  AdaBoostRegressor... done (43.6s)
  LinearRegression... done (0.4s)
  DecisionTreeRegressor... done (0.8s)
  MLPRegressor... done (98.3s)

Algorithm                 Default CV   Default Test  Quick CV     Quick Test   Full CV      Full Test    Time(s)   
-------------------------------------------------------------------------------------------------------------------
SVR                       0.1122       0.1821        0.4446       0.4585       0.4605       0.4617       1.5       
KNeighborsRegressor       0.3172       0.4302        0.4039       0.4320       0.4070       0.4303       0.8       
RandomForestRegressor     0.3909       0.4428        0.4272       0.4763       0.4297       0.4676       38.4      
AdaBoostRegressor         0.4030       0.4301        0.4171       0.4531       0.4223       0.4702       43.6      
LinearRegression          0.4493       0.4526        0.4493       0.4526       0.4493       0.4526       0.4       
DecisionTreeRegressor     -0.1328      0.0607        0.3516       0.2949       0.3516       0.2949       0.8       
MLPRegressor              0.3745       0.4005        0.4520       0.4710       0.4520       0.4710       98.3      

-------------------------------------------------------------------------------------------------------------------
Legend:
  Default CV/Test   = scikit-learn with default hyperparameters
  Quick CV/Test     = OptuML with 20 trials (timeout 60s)
  Full CV/Test      = OptuML with 50 trials (timeout 60s)
  CV                = mean cross-validation score on training set
  Test              = score on held-out test set
  Time(s)           = total wall time (default + quick + full)
```