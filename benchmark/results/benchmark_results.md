# OptuML Benchmark Results

Comparing default scikit-learn hyperparameters against OptuML optimization (20 quick / 50 full trials).

**Columns:** `CV` = mean cross-validation score on training set; `Test` = score on held-out test set; `Time` = total wall time (default + quick + full).

## Iris (Classification) — Accuracy

| Algorithm | Default CV | Default Test | Quick CV | Quick Test | Full CV | Full Test | Time (s) |
|-----------|:----------:|:------------:|:--------:|:----------:|:-------:|:---------:|:--------:|
| SVC | 0.9750 | 0.9667 | 0.9833 | 1.0000 | 0.9833 | 1.0000 | 1.8 |
| KNeighborsClassifier | 0.9750 | 1.0000 | 0.9750 | 0.9333 | 0.9750 | 0.9333 | 0.7 |
| RandomForestClassifier | 0.9500 | 0.9000 | 0.9667 | 0.9333 | 0.9667 | 0.9333 | 32.5 |
| ExtraTreesClassifier | 0.9583 | 0.9333 | 0.9667 | 0.9667 | 0.9750 | 0.9333 | 19.1 |
| AdaBoostClassifier | 0.9500 | 0.9333 | 0.9583 | 0.9333 | 0.9583 | 0.9333 | 35.2 |
| GradientBoostingClassifier | 0.9667 | 0.9667 | 0.9667 | 0.9667 | 0.9667 | 0.9667 | 101.2 |
| HistGradientBoostingClassifier | 0.9417 | 0.9000 | 0.9667 | 0.9000 | 0.9667 | 0.9000 | 49.3 |
| LogisticRegression | 0.9667 | 0.9667 | 0.9750 | 1.0000 | 0.9750 | 1.0000 | 2.7 |
| RidgeClassifier | 0.8583 | 0.8000 | 0.8833 | 0.7667 | 0.8833 | 0.7667 | 0.8 |
| DecisionTreeClassifier | 0.9417 | 0.9333 | 0.9667 | 0.9333 | 0.9667 | 0.9333 | 0.7 |
| GaussianNB | 0.9583 | 0.9667 | 0.9583 | 0.9667 | 0.9583 | 0.9667 | 0.6 |
| QDA | 0.9750 | 1.0000 | 0.9750 | 1.0000 | 0.9750 | 1.0000 | 0.5 |
| MLPClassifier | 0.9667 | 1.0000 | 0.9833 | 0.9333 | 0.9833 | 0.9333 | 47.2 |
| SGDClassifier | 0.7833 | 0.6667 | 0.9667 | 0.9000 | 0.9667 | 0.9000 | 1.2 |
| XGBClassifier | 0.9500 | 0.9333 | 0.9667 | 0.9667 | 0.9667 | 0.9667 | 85.0 |
| LGBMClassifier | 0.9583 | 0.9000 | 0.9667 | 0.9000 | 0.9750 | 0.9333 | 7.8 |

## Wine (Classification) — Accuracy

| Algorithm | Default CV | Default Test | Quick CV | Quick Test | Full CV | Full Test | Time (s) |
|-----------|:----------:|:------------:|:--------:|:----------:|:-------:|:---------:|:--------:|
| SVC | 0.6480 | 0.6944 | 0.9793 | 0.9444 | 0.9793 | 0.9444 | 58.3 |
| KNeighborsClassifier | 0.7108 | 0.8056 | 0.7899 | 0.8611 | 0.7899 | 0.8611 | 0.7 |
| RandomForestClassifier | 0.9862 | 1.0000 | 0.9862 | 1.0000 | 0.9862 | 1.0000 | 28.4 |
| ExtraTreesClassifier | 0.9791 | 1.0000 | 0.9862 | 1.0000 | 0.9862 | 1.0000 | 26.6 |
| AdaBoostClassifier | 0.9648 | 0.9167 | 0.9443 | 0.9722 | 0.9722 | 0.9722 | 45.9 |
| GradientBoostingClassifier | 0.9584 | 0.9444 | 0.9931 | 1.0000 | 0.9931 | 1.0000 | 103.4 |
| HistGradientBoostingClassifier | 0.9581 | 1.0000 | 0.9791 | 1.0000 | 0.9791 | 1.0000 | 49.9 |
| LogisticRegression | 0.9581 | 0.9722 | 0.9724 | 0.9722 | 0.9724 | 0.9722 | 32.1 |
| RidgeClassifier | 0.9931 | 0.9722 | 0.9931 | 0.9722 | 0.9931 | 0.9722 | 0.8 |
| DecisionTreeClassifier | 0.9163 | 0.9444 | 0.8951 | 0.8611 | 0.9232 | 0.9444 | 0.8 |
| GaussianNB | 0.9719 | 0.9722 | 0.9791 | 1.0000 | 0.9791 | 1.0000 | 0.6 |
| QDA | 0.9788 | 1.0000 | 0.9931 | 1.0000 | 0.9931 | 1.0000 | 0.6 |
| MLPClassifier | 0.9512 | 0.9444 | 0.9443 | 0.9722 | 0.9512 | 0.9722 | 91.2 |
| SGDClassifier | 0.4980 | 0.7500 | 0.7268 | 0.7222 | 0.7483 | 0.7500 | 2.5 |
| XGBClassifier | 0.9443 | 1.0000 | 0.9862 | 1.0000 | 0.9931 | 1.0000 | 85.3 |
| LGBMClassifier | 0.9722 | 1.0000 | 0.9862 | 1.0000 | 0.9862 | 1.0000 | 12.2 |

## Breast Cancer (Classification) — Accuracy

| Algorithm | Default CV | Default Test | Quick CV | Quick Test | Full CV | Full Test | Time (s) |
|-----------|:----------:|:------------:|:--------:|:----------:|:-------:|:---------:|:--------:|
| SVC | 0.9099 | 0.9298 | 0.9516 | 0.9649 | 0.9516 | 0.9649 | 171.5 |
| KNeighborsClassifier | 0.9363 | 0.9123 | 0.9407 | 0.9474 | 0.9429 | 0.9474 | 0.9 |
| RandomForestClassifier | 0.9538 | 0.9561 | 0.9582 | 0.9474 | 0.9582 | 0.9474 | 37.7 |
| ExtraTreesClassifier | 0.9714 | 0.9561 | 0.9736 | 0.9561 | 0.9758 | 0.9561 | 27.4 |
| AdaBoostClassifier | 0.9692 | 0.9561 | 0.9758 | 0.9561 | 0.9780 | 0.9561 | 70.2 |
| GradientBoostingClassifier | 0.9560 | 0.9561 | 0.9780 | 0.9737 | 0.9780 | 0.9737 | 107.6 |
| HistGradientBoostingClassifier | 0.9714 | 0.9737 | 0.9802 | 0.9561 | 0.9802 | 0.9561 | 44.5 |
| LogisticRegression | 0.9429 | 0.9649 | 0.9582 | 0.9737 | 0.9604 | 0.9737 | 28.6 |
| RidgeClassifier | 0.9538 | 0.9561 | 0.9670 | 0.9561 | 0.9670 | 0.9561 | 0.8 |
| DecisionTreeClassifier | 0.9099 | 0.9123 | 0.9341 | 0.9561 | 0.9363 | 0.9123 | 1.4 |
| GaussianNB | 0.9363 | 0.9386 | 0.9407 | 0.9386 | 0.9407 | 0.9386 | 0.6 |
| QDA | --- | --- | 0.9582 | 0.9649 | 0.9582 | 0.9649 | 0.6 |
| MLPClassifier | 0.9407 | 0.9211 | 0.9451 | 0.9561 | 0.9451 | 0.9561 | 129.6 |
| SGDClassifier | 0.7626 | 0.9211 | 0.9231 | 0.9211 | 0.9231 | 0.9211 | 1.4 |
| XGBClassifier | 0.9626 | 0.9561 | 0.9780 | 0.9474 | 0.9780 | 0.9474 | 38.6 |
| LGBMClassifier | 0.9714 | 0.9649 | 0.9780 | 0.9649 | 0.9780 | 0.9649 | 9.0 |

## Diabetes (Regression) — R²

| Algorithm | Default CV | Default Test | Quick CV | Quick Test | Full CV | Full Test | Time (s) |
|-----------|:----------:|:------------:|:--------:|:----------:|:-------:|:---------:|:--------:|
| SVR | 0.1122 | 0.1821 | 0.4446 | 0.4585 | 0.4605 | 0.4617 | 1.4 |
| KNeighborsRegressor | 0.3172 | 0.4302 | 0.4039 | 0.4320 | 0.4070 | 0.4303 | 0.7 |
| RandomForestRegressor | 0.3909 | 0.4428 | 0.4274 | 0.4706 | 0.4291 | 0.4663 | 33.4 |
| ExtraTreesRegressor | 0.4064 | 0.4685 | 0.4492 | 0.5076 | 0.4492 | 0.5076 | 17.5 |
| AdaBoostRegressor | 0.4030 | 0.4301 | 0.4240 | 0.4409 | 0.4240 | 0.4409 | 48.6 |
| GradientBoostingRegressor | 0.3610 | 0.4529 | 0.4373 | 0.4731 | 0.4472 | 0.4880 | 47.1 |
| HistGradientBoostingRegressor | 0.3422 | 0.3752 | 0.4350 | 0.4802 | 0.4393 | 0.4718 | 38.7 |
| LinearRegression | 0.4493 | 0.4526 | 0.4493 | 0.4526 | 0.4493 | 0.4526 | 0.0 |
| Ridge | 0.3802 | 0.4192 | 0.4565 | 0.4608 | 0.4565 | 0.4608 | 0.5 |
| Lasso | 0.3238 | 0.3576 | 0.4555 | 0.4712 | 0.4557 | 0.4716 | 0.5 |
| ElasticNet | -0.0216 | -0.0025 | 0.4564 | 0.4607 | 0.4565 | 0.4615 | 0.5 |
| DecisionTreeRegressor | -0.1328 | 0.0607 | 0.3000 | 0.1769 | 0.3001 | 0.2227 | 0.8 |
| MLPRegressor | 0.3862 | 0.4012 | 0.4520 | 0.4710 | 0.4520 | 0.4710 | 97.0 |
| SGDRegressor | 0.3671 | 0.4126 | 0.4575 | 0.4597 | 0.4585 | 0.4581 | 3.6 |
| XGBRegressor | 0.2398 | 0.3675 | 0.4290 | 0.4745 | 0.4290 | 0.4745 | 94.5 |
| LGBMRegressor | 0.3405 | 0.3954 | 0.4281 | 0.4621 | 0.4310 | 0.4635 | 9.8 |
