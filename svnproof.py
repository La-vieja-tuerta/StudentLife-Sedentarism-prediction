LinearSVC(C=1000, verbose=True)
                precision    recall  f1-score   support
less sedentary       0.90      0.90      0.90      5261
     sedentary       0.00      0.00      0.00      1961
very sedentary       0.72      0.98      0.83      5227
   avg / total       0.68      0.79      0.73     12449


clf = SGDClassifier(max_iter=10000)
                precision    recall  f1-score   support
less sedentary       0.91      0.90      0.90      5313
     sedentary       0.00      0.00      0.00      1964
very sedentary       0.71      0.99      0.82      5172
   avg / total       0.68      0.79      0.73     12449